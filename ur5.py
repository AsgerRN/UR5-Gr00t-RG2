import numpy as np
from typing import Any
from numpy.typing import NDArray
from rtde_control import RTDEControlInterface  # type: ignore
from rtde_receive import RTDEReceiveInterface  # type: ignore
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
from gripper import UR5_RG2


class UR5:
    """
    UR5e interface.

    State  (7): [j0..j5, gripper_open in 0..1]
    Action (7): [j0..j5 absolute joint targets, gripper_open in 0..1]
    Camera    : HxWx3 uint8, RGB
    """

    def __init__(self, ur_ip: str):
        self.c: Any = RTDEControlInterface(ur_ip)
        self.r: Any = RTDEReceiveInterface(ur_ip)
        self.gripper = UR5_RG2(ur_ip)
        self.camera_pipeline = self._init_camera()

    def __del__(self):
        if getattr(self, "c", None) is not None:
            self.c.disconnect()
        if getattr(self, "r", None) is not None:
            self.r.disconnect()
        if getattr(self, "camera_pipeline", None) is not None:
            self.camera_pipeline.stop()

    @property
    def pose(self) -> NDArray[np.float32]:
        """[j0..j5, gripper_open]"""
        joints = self.r.getActualQ()
        gripper = float(np.clip(self.gripper.get_width() / 110.0, 0.0, 1.0))
        return np.array([*joints, gripper], dtype=np.float32)

    def move(self, action: NDArray[np.float64 | np.float32], max_joint_step: float | None = None) -> None:
        """
        Execute [j0..j5 absolute targets, gripper_open] via servoJ (50 Hz stream).

        max_joint_step: if set (rad), each joint target is clipped to within this
        distance of the current joint position. Use ~0.05 for first model runs.
        """
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")

        target_joints = np.asarray(action[:6], dtype=np.float64)
        if max_joint_step is not None:
            current = np.asarray(self.r.getActualQ(), dtype=np.float64)
            target_joints = current + np.clip(target_joints - current, -max_joint_step, max_joint_step)

        self.gripper.overwrite_move(action[6] * 110.0)

        # servoJ(q, velocity, acceleration, dt, lookahead_time, gain)
        self.c.servoJ(target_joints.tolist(), 1.0, 1.0, 0.02, 0.1, 300)

    def _init_camera(self) -> Pipeline:
        pipeline = Pipeline()
        config = Config()

        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 60)
        config.enable_stream(color_profile)
        pipeline.start(config)
        print("[INFO] Orbbec Gemini 305 initialized via SDK at 640x480 @ 60FPS")

        return pipeline

    @property
    def camera(self) -> NDArray[np.uint8]:
        """HxWx3 uint8 RGB, C-contiguous copy."""
        for _ in range(4):
            frames = self.camera_pipeline.wait_for_frames(500)
            if frames:
                break
            print("⚠️ Waiting for camera frame...")
        else:
            raise RuntimeError("Failed to retrieve frames from Orbbec camera")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("No color frame in frameset")

        data = color_frame.get_data()
        frame = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape((color_frame.get_height(), color_frame.get_width(), 3))
            .copy()
        )
        return np.ascontiguousarray(frame)


class UR5_Dummy(UR5):
    """Camera only, no robot."""

    def __init__(self, _: str):
        self.c = None
        self.r = None
        self.gripper = None
        self.camera_pipeline = self._init_camera()

    @property
    def pose(self) -> NDArray[np.float32]:
        return np.zeros((7,), dtype=np.float32)

    def move(self, action: NDArray[np.float64 | np.float32], max_joint_step: float | None = None) -> None:
        print("Dummy action:", action)


if __name__ == "__main__":
    # Colour sanity check: hold a RED object in view.
    # cv2.imwrite expects BGR, so convert the RGB frame for saving.
    import cv2

    ur = UR5_Dummy("")
    frame = ur.camera
    print("shape:", frame.shape, "dtype:", frame.dtype, "contiguous:", frame.flags["C_CONTIGUOUS"])
    cv2.imwrite("camera_check.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
