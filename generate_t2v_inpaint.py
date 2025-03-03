import moviepy.editor as mpy
import os
import cv2
import numpy as np
import torch
import requests
from PIL import Image, ImageFilter

from wan.text2video_inpaint import WanT2VInpaint


def download_video(url: str, local_path: str | None = None) -> str:
    if local_path and os.path.exists(local_path):
        print(f"Video already exists at {local_path}")
        return local_path

    print(f"Downloading video from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    if local_path is None:
        import tempfile

        _, local_path = tempfile.mkstemp(suffix=".mp4")

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved video to {local_path}")
    return local_path


def load_video_frames(
    video_path: str, max_frames: int | None = None
) -> list[np.ndarray]:
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


def video_frames_to_pil(frames: list[np.ndarray]) -> list[Image.Image]:
    return [Image.fromarray(frame) for frame in frames]


def generate_t2v_inpaint(
    model_config,
    checkpoint_dir: str,
    prompt: str,
    device_id: int = 0,
    input_video_url: str = "https://replicate.delivery/xezq/UpqkYwl7hTqTJ5rfIE0eb5WpDv6quG6PeFcdtH157Y2uonloA/tmp2dc62s6m.output.mp4",
    mask_video_url: str = "https://replicate.delivery/xezq/N94eQyo3WBVWXyN8Aj4kpwQ42l6prb578m9tCJWaFgREnlJKA/masked_video.mp4",
    strength: float = 0.8,
    guide_scale: float = 5.0,
    sampling_steps: int = 50,
    negative_prompt: str = "",
    seed: int = 42,
    save_path: str = "inpainted_output.mp4",
    model=None,
) -> torch.Tensor:
    input_video_path = download_video(input_video_url)
    mask_video_path = download_video(mask_video_url)

    input_frames = load_video_frames(input_video_path)
    mask_frames = load_video_frames(mask_video_path)

    min_frames = min(len(input_frames), len(mask_frames))
    input_frames = input_frames[:min_frames]
    mask_frames = mask_frames[:min_frames]

    input_pil_frames = video_frames_to_pil(input_frames)
    mask_pil_frames = video_frames_to_pil(mask_frames)

    frame_height, frame_width = input_frames[0].shape[:2]

    print(f"Initializing WanT2VInpaint model...")
    if model is None:
        model = WanT2VInpaint(
            config=model_config, checkpoint_dir=checkpoint_dir, device_id=device_id
        )

    print(f"Running inpainting with prompt: '{prompt}'")
    result = model.generate_inpaint(
        input_prompt=prompt,
        init_video=input_pil_frames,
        mask=mask_pil_frames,
        strength=strength,
        size=(frame_width, frame_height),
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        n_prompt=negative_prompt,
        seed=seed,
        offload_model=True,
    )

    if save_path:
        save_video_tensor(result, save_path, fps=30)
        print(f"Saved output video to {save_path}")

    return result


def save_video_tensor(video_tensor, output_path, fps=30):
    """
    Saves a video tensor as an MP4 file using moviepy for better compatibility.

    Args:
        video_tensor: Tensor of shape [C, F, H, W]
        output_path: Path to save the video
        fps: Frames per second
    """
    if torch.is_tensor(video_tensor):
        video = video_tensor.detach().cpu().numpy()
        video = np.transpose(video, (1, 2, 3, 0))
        video = (video * 255).astype(np.uint8)
    else:
        video = video_tensor

    clip = mpy.ImageSequenceClip([frame for frame in video], fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps)

    print(f"Saved video to {output_path}")
