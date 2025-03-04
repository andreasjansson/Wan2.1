import os
import subprocess
import time
import random
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
import moviepy.editor as mpy
from cog import BasePredictor, Input, Path

from wan.configs import WAN_CONFIGS
from wan.text2video_inpaint import WanT2VInpaint
from distributed import DistributedManager

MODEL_ROOT_DIR = Path("/weights")
WEIGHTS_DIR = MODEL_ROOT_DIR / "Wan2.1-T2V-1.3B"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/wan2.1/model_cache/Wan2.1-T2V-1.3B.tar"
)


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if not WEIGHTS_DIR.exists():
            download_weights(MODEL_URL, MODEL_ROOT_DIR)

        # Define model arguments
        model_args = {
            'config': WAN_CONFIGS["t2v-1.3B"],
            'checkpoint_dir': str(WEIGHTS_DIR),
            # device_id, rank, dit_fsdp, t5_fsdp, and use_usp will be set by the manager
        }

        # Set up distributed processing with the model class and arguments
        self.manager = DistributedManager(WanT2VInpaint, model_args)
        self.manager.setup()

    def predict(
        self,
        prompt: str = Input(description="Prompt for inpainting the masked area"),
        input_video: Path = Input(description="Original video to be inpainted"),
        mask_video: Path = Input(
            description="Mask video (white areas will be inpainted). Leave blank for video-to-video",
            default=None,
        ),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        strength: float = Input(
            description="Strength of inpainting effect, 1.0 is full regeneration",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        guide_scale: float = Input(
            description="Guidance scale for prompt adherence",
            ge=1.0,
            le=15.0,
            default=5.0,
        ),
        sampling_steps: int = Input(
            description="Number of sampling steps", ge=20, le=100, default=50
        ),
        inpaint_fixup_steps: int = Input(
            description="Number of steps for final inpaint fixup. Ignored when in video-to-video mode (when mask_video is empty)",
            ge=0,
            le=10,
            default=0,
        ),
        expand_mask: int = Input(
            description="Expand the mask by a number of pixels",
            ge=0,
            le=100,
            default=10,
        ),
        frames_per_second: int = Input(
            description="Output video FPS", ge=5, le=30, default=16
        ),
        seed: int = Input(
            description="Random seed. Leave blank for random", default=-1
        ),
    ) -> Path:
        seed = seed_or_random_seed(seed)

        input_frames = load_video_frames(str(input_video))
        if mask_video:
            mask_frames = load_video_frames(str(mask_video))
        else:
            mask_frames = []
            for frame in input_frames:
                white_frame = np.ones_like(frame) * 255
                mask_frames.append(white_frame)

        min_frames = min(len(input_frames), len(mask_frames))
        input_frames = input_frames[:min_frames]
        mask_frames = mask_frames[:min_frames]

        input_pil_frames = [Image.fromarray(frame) for frame in input_frames]
        mask_pil_frames = [Image.fromarray(frame) for frame in mask_frames]
        mask_pil_frames = expand_mask_frames(mask_pil_frames, blur_radius=expand_mask)

        frame_height, frame_width = input_frames[0].shape[:2]

        # Prepare parameters for the task
        params = {
            "input_prompt": prompt,
            "init_video": input_pil_frames,
            "mask": mask_pil_frames,
            "size": (frame_width, frame_height),
            "strength": strength,
            "sampling_steps": sampling_steps,
            "inpaint_fixup_steps": inpaint_fixup_steps if mask_video else 0,
            "guide_scale": guide_scale,
            "n_prompt": negative_prompt,
            "seed": seed,
            "offload_model": False,
        }

        # Submit task and get result
        result = self.manager.submit_task(params)

        # Save the result
        output_path = Path("output.mp4")
        save_video_tensor(result, str(output_path), fps=frames_per_second)

        return output_path

    def __del__(self):
        """Clean up resources when the predictor is destroyed"""
        if hasattr(self, 'manager'):
            self.manager.cleanup()


def download_weights(url: str, dest: Path):
    """Download model weights from a URL"""
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


def load_video_frames(
    video_path: str, max_frames: int = None
) -> list[np.ndarray]:
    """Load frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if max_frames and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def expand_mask_frames(
    mask_pil_frames: list[Image.Image], blur_radius: int = 7
) -> list[Image.Image]:
    """
    Expands the mask using PIL's blur filter and thresholding.
    """
    expanded_masks = []

    for mask in mask_pil_frames:
        if mask.mode != "L":
            gray_mask = mask.convert("L")
        else:
            gray_mask = mask.copy()

        blurred = gray_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        expanded = blurred.point(lambda p: 255 if p > 0 else 0, mode="1").convert("L")

        if mask.mode != "L":
            expanded = expanded.convert(mask.mode)

        expanded_masks.append(expanded)

    return expanded_masks


def save_video_tensor(video_tensor, output_path, fps=30):
    """Save video tensor to MP4 file"""
    if torch.is_tensor(video_tensor):
        video = video_tensor.detach().cpu().numpy()
        video = np.transpose(video, (1, 2, 3, 0))
        video = (video * 255).astype(np.uint8)
    else:
        video = video_tensor

    clip = mpy.ImageSequenceClip([frame for frame in video], fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps)

    print(f"Saved video to {output_path}")


def seed_or_random_seed(seed: int = None) -> int:
    # Max seed is 2147483647
    if not seed or seed <= 0:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

    print(f"Using seed: {seed}\n")
    return seed
