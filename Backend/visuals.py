"""
AI-generated visuals: one image per scene, animated with a slow zoom/pan
("Ken Burns") and cut to match the narration.

Image providers are tried in order (IMAGE_PROVIDERS, default
"pollinations,cloudflare,local"). A provider that fails twice in a row is
skipped for the rest of the video, so a slow or broken service doesn't stall
generation.
"""

import base64
import io
import os
import random
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import imageio_ffmpeg
import requests
from PIL import Image, ImageFilter

from logstream import log
from utils import TEMP_DIR

WIDTH, HEIGHT = 1080, 1920
FPS = 30

IMAGE_STYLE = os.getenv(
    "IMAGE_STYLE",
    "cinematic, clean, photorealistic, dramatic lighting, shallow depth of field, "
    "sharp focus, highly detailed, 35mm film still, vertical composition",
)
NEGATIVE = "no text, no words, no letters, no watermark, no logo"

_local_pipe = None


class ProviderError(Exception):
    pass


def _pollinations(prompt: str, size=(WIDTH, HEIGHT)) -> bytes:
    key = os.getenv("POLLINATIONS_API_KEY", "").strip()
    if not key:
        raise ProviderError("POLLINATIONS_API_KEY not set")

    models = [m.strip() for m in os.getenv("POLLINATIONS_MODELS", "zimage,flux").split(",") if m.strip()]
    last_error = None
    for model in models:
        try:
            response = requests.get(
                f"https://gen.pollinations.ai/image/{quote(prompt)}",
                params={
                    "model": model,
                    "width": size[0],
                    "height": size[1],
                    "nologo": "true",
                    "seed": random.randint(1, 10**9),
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=120,
            )
            if response.ok and response.headers.get("content-type", "").startswith("image/"):
                return response.content
            last_error = f"{model}: HTTP {response.status_code} {response.text[:200]}"
        except requests.RequestException as err:
            last_error = f"{model}: {err}"
    raise ProviderError(last_error)


def _cloudflare(prompt: str, size=(WIDTH, HEIGHT)) -> bytes:
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise ProviderError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set")

    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/"
        "@cf/black-forest-labs/flux-1-schnell",
        headers={"Authorization": f"Bearer {token}"},
        json={"prompt": prompt, "steps": 8},
        timeout=120,
    )
    if not response.ok:
        raise ProviderError(f"HTTP {response.status_code} {response.text[:200]}")
    image_b64 = response.json().get("result", {}).get("image")
    if not image_b64:
        raise ProviderError(f"No image in response: {response.text[:200]}")
    return base64.b64decode(image_b64)


def _local(prompt: str, size=(WIDTH, HEIGHT)) -> bytes:
    """Stable Diffusion Turbo on the local CPU (needs torch + diffusers)."""
    global _local_pipe
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as err:
        raise ProviderError(f"torch/diffusers not installed: {err}")

    if _local_pipe is None:
        log("[*] Loading local Stable Diffusion model (first run downloads ~2.5 GB)...", "info")
        _local_pipe = AutoPipelineForText2Image.from_pretrained(
            os.getenv("LOCAL_SD_MODEL", "stabilityai/sd-turbo"), torch_dtype=torch.float32
        )
        _local_pipe.to("cpu")

    image = _local_pipe(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=0.0,
        width=512 if size[0] < size[1] else 896,
        height=896 if size[0] < size[1] else 512,
    ).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


PROVIDERS = {"pollinations": _pollinations, "cloudflare": _cloudflare, "local": _local}


def _cover(image: Image.Image, size=(WIDTH, HEIGHT)) -> Image.Image:
    """Scale and center-crop an image to fill the frame."""
    width, height = size
    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
    )
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    # Upscaled images (e.g. 1024px Cloudflare output in a 1080p frame) look soft;
    # a light unsharp mask restores edge detail without halos
    if scale > 1.05:
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))
    return image


class ImageGenerator:
    def __init__(self, on_log=None, size=(WIDTH, HEIGHT)) -> None:
        self.size = size
        order = os.getenv("IMAGE_PROVIDERS", "pollinations,cloudflare,local")
        self.providers = [p.strip() for p in order.split(",") if p.strip() in PROVIDERS]
        self.failures = {p: 0 for p in self.providers}
        self.on_log = on_log or (lambda message, level: log(message, level))

    def generate(self, prompt: str, path: str) -> Optional[str]:
        """Create one image at `path`. Returns the provider used, or None."""
        full_prompt = f"{prompt}. {IMAGE_STYLE}. {NEGATIVE}"
        for provider in self.providers:
            if self.failures[provider] >= 2:
                continue
            started = time.time()
            try:
                data = PROVIDERS[provider](full_prompt, self.size)
                _cover(Image.open(io.BytesIO(data)), self.size).save(path, "JPEG", quality=92)
                self.failures[provider] = 0
                self.on_log(
                    f"   [{provider}] image ready in {time.time() - started:.0f}s", "info"
                )
                return provider
            except Exception as err:
                self.failures[provider] += 1
                self.on_log(f"   [{provider}] failed: {str(err)[:200]}", "warning")
        return None


def group_scenes(sentences: List[str], durations: List[float], target: float = 3.5):
    """
    Build one scene per sentence so each image matches the words being spoken.
    Sentences longer than `target` seconds are split at commas/conjunctions into
    smaller phrases, with time shared out by word count. Very short sentences
    (under 1.2 s, e.g. "Why?") are merged into the next one.
    """
    scenes = []
    carry_text, carry_time = "", 0.0
    for sentence, duration in zip(sentences, durations):
        text = f"{carry_text} {sentence}".strip()
        duration += carry_time
        if duration < 1.2:
            carry_text, carry_time = text, duration
            continue
        carry_text, carry_time = "", 0.0

        parts = [text]
        if duration > target:
            pieces = [p.strip() for p in re.split(r",\s+|;\s+|\s+(?=(?:but|and|so|because|while|which)\s)", text) if p.strip()]
            # Merge pieces until each is at least ~2 seconds of speech
            words_total = max(1, len(text.split()))
            parts, current = [], ""
            for piece in pieces:
                current = f"{current} {piece}".strip()
                if len(current.split()) / words_total * duration >= 2.0:
                    parts.append(current)
                    current = ""
            if current:
                if parts:
                    parts[-1] = f"{parts[-1]} {current}"
                else:
                    parts.append(current)

        words_total = max(1, sum(len(p.split()) for p in parts))
        for part in parts:
            scenes.append((part, duration * len(part.split()) / words_total))

    if carry_text:
        if scenes:
            text, d = scenes.pop()
            scenes.append((f"{text} {carry_text}", d + carry_time))
        else:
            scenes.append((carry_text, carry_time))
    return scenes


def _ken_burns_segment(
    image_path: str, duration: float, index: int, out_path: str, size=(WIDTH, HEIGHT)
) -> None:
    frames = max(1, round(duration * FPS))
    zoom_step = 0.12 / frames
    # Alternate zoom in / zoom out and drift direction for variety
    if index % 2 == 0:
        zoom = f"min(1+{zoom_step}*on,1.12)"
    else:
        zoom = f"max(1.12-{zoom_step}*on,1)"
    x = "iw/2-(iw/zoom/2)" if index % 3 == 0 else (
        "(iw-iw/zoom)*on/{f}".format(f=frames) if index % 3 == 1 else "(iw-iw/zoom)*(1-on/{f})".format(f=frames)
    )
    y = "ih/2-(ih/zoom/2)"
    vf = (
        # Lanczos supersampling keeps the slow zoom sharp
        f"scale={size[0] * 2}:{size[1] * 2}:flags=lanczos,"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={size[0]}x{size[1]}:fps={FPS},"
        "format=yuv420p"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-loop", "1", "-i", image_path,
            "-vf", vf, "-frames:v", str(frames),
            # Near-lossless intermediate; the final pass does the real compression
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            out_path,
        ],
        check=True,
    )


def build_ai_video(
    scenes: List[tuple],
    prompts: List[str],
    on_log=None,
    min_success_ratio: float = 0.6,
    size=(WIDTH, HEIGHT),
) -> str:
    """
    Generate one image per scene and render them into a silent 1080x1920
    video (default 1080x1920) whose length matches the narration.

    Raises RuntimeError if too few images could be generated, so the caller
    can fall back to stock footage.
    """
    emit = on_log or (lambda message, level: log(message, level))
    generator = ImageGenerator(on_log=emit, size=size)
    work_dir = Path(TEMP_DIR) / f"ai_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Optional[str]] = []
    for i, prompt in enumerate(prompts):
        emit(f"[+] Image {i + 1}/{len(prompts)}: {prompt[:90]}", "info")
        path = str(work_dir / f"scene_{i:02d}.jpg")
        image_paths.append(path if generator.generate(prompt, path) else None)

    made = sum(1 for p in image_paths if p)
    if made < max(1, int(len(prompts) * min_success_ratio)):
        raise RuntimeError(f"Only {made}/{len(prompts)} images could be generated.")

    # Reuse the previous image for any scene whose image failed
    last = next(p for p in image_paths if p)
    for i, path in enumerate(image_paths):
        if path:
            last = path
        else:
            image_paths[i] = last

    emit("[+] Animating images...", "info")
    segment_list = work_dir / "segments.txt"
    with open(segment_list, "w", encoding="utf-8") as f:
        for i, ((_, duration), image_path) in enumerate(zip(scenes, image_paths)):
            segment = str(work_dir / f"segment_{i:02d}.mp4")
            _ken_burns_segment(image_path, duration, i, segment, size)
            f.write(f"file '{Path(segment).as_posix()}'\n")

    output = str(Path(TEMP_DIR) / f"{uuid.uuid4()}.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(segment_list),
            "-c", "copy", output,
        ],
        check=True,
    )
    return output
