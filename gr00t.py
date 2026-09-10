import torch
import numpy as np
from typing import Any
from numpy.typing import NDArray
from lerobot.policies.groot.modeling_groot import GrootPolicy # type: ignore
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors # type: ignore
from huggingface_hub import snapshot_download # type: ignore


class Gr00t:

    def __init__(self, checkpoint_name:str = "nvidia/GR00T-N1.7-SimplerEnv-Bridge"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy, self.preprocessor, self.postprocessor = self._init_policy(checkpoint_name)

        for step in self.preprocessor.steps:
            print(type(step).__name__)

            if hasattr(step, "video_modality_keys"):
                print(
                    "video_modality_keys:",
                    step.video_modality_keys,
                )

            if hasattr(step, "video_horizon"):
                print(
                    "video_horizon:",
                    step.video_horizon,
                )

            if hasattr(step, "image_target_size"):
                print(
                    "image_target_size:",
                    step.image_target_size,
                )

            if hasattr(step, "image_crop_size"):
                print(
                    "image_crop_size:",
                    step.image_crop_size,
                )

            if hasattr(step, "crop_fraction"):
                print(
                    "crop_fraction:",
                    step.crop_fraction,
                )

    def _init_policy(self, checkpoint_name:str) -> tuple[Any, Any, Any]:
        checkpoint_metadata_path = snapshot_download(
            checkpoint_name,
            allow_patterns=[
                "config.json",
                "processor_config.json",
                "statistics.json",
                "embodiment_id.json",
            ],
        )

        policy = GrootPolicy.from_pretrained(checkpoint_name, device=self.device) # type: ignore
        
        policy.config.base_model_path = checkpoint_metadata_path
        policy.config.embodiment_tag = "simpler_env_widowx"
        policy = policy.to(self.device)
        policy.eval()

        preprocessor, postprocessor = make_groot_pre_post_processors(config=policy.config)

        return policy, preprocessor, postprocessor
        

    def predict_action(self, camera_frame: NDArray[np.uint8], robot_state: NDArray[np.float32], text_instruction: str) -> NDArray[np.float32]:
        if camera_frame.ndim != 3 or camera_frame.shape[2] != 3:
            raise ValueError(f"Expected RGB HWC image, got shape {camera_frame.shape}")
        if robot_state.shape != (8,):
            raise ValueError(f"Expected GR00T state (8,), got {robot_state.shape}: {robot_state}")

        print("RAW ROBOT STATE:")
        print(robot_state)
        print("shape:", robot_state.shape)

        if robot_state.shape != (8,):
            raise ValueError(
                f"Expected GR00T WidowX robot_state shape (8,), got {robot_state.shape}: {robot_state}"
    )

        # The LeRobot Groot processor expects a C,H,W Torch image tensor.
        image = torch.from_numpy(np.ascontiguousarray(camera_frame.transpose(2, 0, 1))) # type: ignore
        observation: dict[str, torch.Tensor | str] = {
            "observation.images.image_0": image,
            "observation.state": torch.as_tensor(robot_state, dtype=torch.float32),
            "task": text_instruction,
        }

        batch = self.preprocessor(observation)
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        with torch.inference_mode():
            raw_action_chunk = self.policy.predict_action_chunk(batch)

        # The postprocessor decodes normalized/padded model actions into robot actions.
        decoded_action_chunk = self.postprocessor(raw_action_chunk)
        return decoded_action_chunk[0].cpu().numpy()

if __name__ == "__main__":
    # Example usage
    gr00t = Gr00t()

    gr00t.predict_action(
        camera_frame=np.zeros((256, 256, 3), dtype=np.uint8),
        robot_state=np.zeros((8,), dtype=np.float32),
        text_instruction="Pick up the red block and place it on the green block.",
    )