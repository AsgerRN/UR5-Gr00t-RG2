import torch
import numpy as np
from numpy.typing import NDArray
from lerobot.policies.groot.modeling_groot import GrootPolicy  # type: ignore
from lerobot.policies.factory import make_pre_post_processors  # type: ignore

ACTION_DIM = 7

class Gr00t:
    """
    Fine-tuned GR00T on the UR5e dataset.

    checkpoint_path: the *pretrained_model* folder of your training run, typically
        /mnt/big/Gr00t_checkpoints/train/ur5e_gr00t_expert_only/checkpoints/last/pretrained_model

    Input : RGB frame (HxWx3 uint8) from UR5.camera, state (7,) from UR5.pose
    Output: chunk (T, 7) of ABSOLUTE joint targets + gripper_open (0..1)
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GrootPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()

        # Load the processors SAVED with the checkpoint so normalization uses your dataset stats.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=checkpoint_path
        )

    def reset(self) -> None:
        self.policy.reset()

    def predict_action_chunk(
        self, camera_frame: NDArray[np.uint8], robot_state: NDArray[np.float32], text_instruction: str
    ) -> NDArray[np.float32]:
        if camera_frame.ndim != 3 or camera_frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got {camera_frame.shape}")
        if robot_state.shape != (7,):
            raise ValueError(f"Expected state (7,), got {robot_state.shape}")

        # Same form the LeRobot dataloader gives in training: float32 CHW in [0, 1].
        image = torch.from_numpy(np.ascontiguousarray(camera_frame)).permute(2, 0, 1).float() / 255.0

        observation: dict[str, torch.Tensor | str] = {
            "observation.images.wrist": image,
            "observation.state": torch.as_tensor(robot_state, dtype=torch.float32),
            "task": text_instruction,
        }

        batch = self.preprocessor(observation)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with torch.inference_mode():
            raw_chunk = self.policy.predict_action_chunk(batch)

        decoded = self.postprocessor(raw_chunk)
        chunk = decoded.detach().cpu().numpy() if torch.is_tensor(decoded) else np.asarray(decoded)
        if chunk.ndim == 3:
            chunk = chunk[0]
        chunk = chunk[:, :ACTION_DIM].astype(np.float32)
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        return chunk


if __name__ == "__main__":
    model = Gr00t("/mnt/big/Gr00t_checkpoints/train/ur5e_gr00t_expert_only/checkpoints/last/pretrained_model")
    out = model.predict_action_chunk(
        np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((7,), dtype=np.float32), "Pick up the red object"
    )
    print("chunk shape:", out.shape)
    print(out[:2])