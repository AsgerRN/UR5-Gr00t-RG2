import time
import numpy as np

from ur5 import UR5_Groot
from gr00t import Gr00t


ROBOT_IP = "192.168.1.103"

SPEED = 0.02
ACCELERATION = 0.02

INSTRUCTION = "Pick up the red sphere."


ur = UR5_Groot(ROBOT_IP)
groot = Gr00t()


try:
    while True:
        # 1. Observe robot after previous motion has completed
        camera_frame = ur.camera

        robot_state = np.asarray(
            ur.pose,
            dtype=np.float32,
        )

        # 2. Predict a fresh action chunk
        chunk = groot.predict_action(
            camera_frame=camera_frame,
            robot_state=robot_state,
            text_instruction=INSTRUCTION,
        )

        # 3. Execute the first actions in the chunk
        for action in chunk[:3]:
            print("Executing action:", action)
            ur.move(
                np.asarray(action, dtype=np.float64),
                speed=SPEED,
                acceleration=ACCELERATION,
            )

except KeyboardInterrupt:
    print("\nStopped.")