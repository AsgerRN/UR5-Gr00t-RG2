import cv2
import numpy as np
from typing import Any
from numpy.typing import NDArray
from rtde_control import RTDEControlInterface  # type: ignore
from rtde_receive import RTDEReceiveInterface  # type: ignore
from gripper import UR5_RG2


class UR5:
    def __init__(self, ur_ip: str):
        self.c: Any = RTDEControlInterface(ur_ip)
        self.r: Any = RTDEReceiveInterface(ur_ip)

        self.gripper = UR5_RG2(ur_ip)
        self.cam = self._init_camera(6)

    def __del__(self):
        if getattr(self, "c", None) is not None:
            self.c.disconnect()
        if getattr(self, "r", None) is not None:
            self.r.disconnect()
        if getattr(self, "cam", None) is not None:
            self.cam.release()

    def _ur_pose(self) -> NDArray[np.float64]:
        """Raw physical UR TCP pose."""
        return np.asarray(self.r.getActualTCPPose(), dtype=np.float64)

    @property
    def pose(self) -> NDArray[np.float32]:
        """Pose in UR base coordinates."""
        return self._ur_pose().astype(np.float32)

    def move(self, pose: NDArray[np.float64], speed: float = 0.02, acceleration: float = 0.02) -> None:
        """Move to an absolute pose in UR base coordinates."""
        if pose.shape != (6,):
            raise ValueError(f"Expected UR pose shape (6,), got {pose.shape}")

        self.c.moveL(pose.tolist(), speed, acceleration)

    def _init_camera(self, camera_index: int) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not camera.isOpened():
            raise RuntimeError(
                f"Failed to open /dev/video{camera_index}"
            )

        success, frame = camera.read()

        if not success:
            camera.release()
            raise RuntimeError(
                f"Failed to read from /dev/video{camera_index}"
            )

        print(
            f"Using /dev/video{camera_index}: "
            f"{frame.shape[1]}x{frame.shape[0]}"
        )

        return camera
    
    @property
    def camera(self) -> NDArray[np.uint8]:
        success, frame_bgr = self.cam.read()

        if not success:
            raise RuntimeError("Failed to read camera frame")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame_rgb, dtype=np.uint8)


class UR5_Groot(UR5):

    BRIDGE_DEFAULT_ROT = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)

    def __init__(self, ur_ip: str):
        super().__init__(ur_ip)

        # UR base -> Bridge base
        self.T_B_U = np.array([
            [-1.0,  0.0, 0.0, -0.44924286],
            [ 0.0, -1.0, 0.0, -0.01044477],
            [ 0.0,  0.0, 1.0, -0.21093492],
            [ 0.0,  0.0, 0.0,  1.0],
        ], dtype=np.float64)

        # UR TCP -> conceptual Bridge EE
        self.T_TCP_E = np.eye(4, dtype=np.float64)

        self.T_U_B = np.linalg.inv(self.T_B_U)
        self.T_E_TCP = np.linalg.inv(self.T_TCP_E)

    @staticmethod
    def _pose6_to_matrix(pose: NDArray[np.float64]) -> NDArray[np.float64]:
        R, _ = cv2.Rodrigues(pose[3:6])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pose[:3]
        return T

    @staticmethod
    def _matrix_to_pose6(T: NDArray[np.float64]) -> NDArray[np.float64]:
        rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=np.float64))
        return np.concatenate([T[:3, 3], rvec.ravel()], dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_static_xyz(R: NDArray[np.float64]) -> NDArray[np.float64]:
        cy = np.hypot(R[0, 0], R[1, 0])
        if cy > 1e-8:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], cy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], cy)
            yaw = 0.0
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @property
    def pose(self) -> NDArray[np.float32]:
        """
        Current pose in GR00T/Bridge state format:

        [x, y, z, roll, pitch, yaw, 0, gripper_open]
        """
        # Important: raw physical UR pose, not self.pose.
        T_U_TCP = self._pose6_to_matrix(self._ur_pose())
        T_B_E = self.T_B_U @ T_U_TCP @ self.T_TCP_E
        position = T_B_E[:3, 3]
        R_groot = T_B_E[:3, :3] @ self.BRIDGE_DEFAULT_ROT.T
        rotation = self._rotation_matrix_to_static_xyz(R_groot)
        gripper_open = float(np.clip(self.gripper.get_width() / 110.0, 0.0, 1.0))
        return np.array([*position, *rotation, 0.0, gripper_open], dtype=np.float32)

    def _action_to_ur_target(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        GR00T action:
            [dx, dy, dz, rx, ry, rz, gripper]

        Returns an absolute physical UR TCP target.
        """
        if action.shape != (7,):
            raise ValueError(f"Expected GR00T action shape (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("GR00T action contains NaN or Inf")

        # Always read the raw UR pose here.
        T_U_TCP = self._pose6_to_matrix(self._ur_pose())
        # Current EE in Bridge coordinates.
        T_B_E = self.T_B_U @ T_U_TCP @ self.T_TCP_E
        # Apply GR00T translation in Bridge base coordinates.
        T_B_E[:3, 3] += action[:3]
        # Apply GR00T rotation in Bridge base coordinates.
        delta_R_B, _ = cv2.Rodrigues(action[3:6])
        T_B_E[:3, :3] = delta_R_B @ T_B_E[:3, :3]
        # Bridge -> UR.
        T_U_TCP_target = self.T_U_B @ T_B_E @ self.T_E_TCP

        return self._matrix_to_pose6(T_U_TCP_target)

    def move(self, pose: NDArray[np.float64], speed: float = 0.02, acceleration: float = 0.02) -> None:
        """
        Execute a GR00T action:

        [dx, dy, dz, rx, ry, rz, gripper]
        """
        if speed <= 0 or acceleration <= 0:
            raise ValueError("speed and acceleration must be positive")

        target_ur_pose = self._action_to_ur_target(pose)
        self.gripper.overwrite_move(np.clip(pose[6] * 110, 0, 100))
        print(f"gripper_goal: {np.clip(pose[6] * 110, 0, 100)} mm, gripper position: {self.gripper.get_width()} mm")
        self.c.moveL(target_ur_pose.tolist(), speed, acceleration)


class UR5_Dummy(UR5):
    def __init__(self, _: str):
        self.c = None
        self.r = None
        self.gripper = None
        self.cam = self._init_camera(6)

    def __del__(self):
        self.cam.release()

    @property
    def pose(self) -> NDArray[np.float32]:
        return np.zeros((8,), dtype=np.float32)

    def move(self, pose: NDArray[np.float64], speed: float = 0.02, acceleration: float = 0.02) -> None:
        print("Dummy pose:", pose)


if __name__ == "__main__":
    """
    ur = UR5_Groot("192.168.1.103")

    print("GR00T pose:")
    print(ur.pose)

    action = np.array([
        0.00,  # ΔX Bridge
        0.00,  # ΔY Bridge
        0.00,  # ΔZ Bridge
        0.00,  # Δ rotation X Bridge
        0.00,  # Δ rotation Y Bridge
        -0.50,  # Δ rotation Z Bridge
        0.00,  # gripper
    ])

    ur.move(
        action,
        speed=0.10,
        acceleration=0.10,
    )
    """


    ur = UR5_Groot("192.168.1.103")

    frame = ur.camera

    print("shape:      ", frame.shape)
    print("dtype:      ", frame.dtype)
    print("min:        ", frame.min())
    print("max:        ", frame.max())
    print("contiguous: ", frame.flags["C_CONTIGUOUS"])

    success, raw_bgr = ur.cam.read()

    if not success:
        raise RuntimeError("Camera read failed")

    cv2.imwrite("camera_raw.png", raw_bgr)

    frame = ur.camera
    cv2.imwrite(
        "camera_groot.png",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
    )