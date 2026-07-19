"""
Convert a track toward folk metal while keeping vocals (and optionally drums) intact.

Pipeline:
  1) Demucs stem split
  2) Riffusion img2img style transfer on non-vocal stems
  3) Remix with original vocals (and drums if --keep-drums)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pydub
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from riffusion.audio_splitter import split_audio
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit.util import get_scheduler


DEFAULT_CHECKPOINT = "riffusion/riffusion-model-v1"
DEFAULT_PROMPT = (
    "folk metal, heavy distorted guitars, folk melodies, tin whistle, "
    "aggressive rhythm guitar, epic metal production"
)
DEFAULT_NEGATIVE = "pop, edm, acoustic only, lo-fi, muffled, speech"


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def load_audio(path: Path) -> pydub.AudioSegment:
    segment = pydub.AudioSegment.from_file(path)
    if segment.frame_rate != 44100:
        segment = segment.set_frame_rate(44100)
    if segment.channels > 2:
        segment = segment.set_channels(2)
    return segment


def overlay_sum(segments: list[pydub.AudioSegment]) -> pydub.AudioSegment:
    out = segments[0]
    for seg in segments[1:]:
        out = out.overlay(seg)
    return out


def scale_image_to_32_stride(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_width = math.ceil(width / 32) * 32
    new_height = math.ceil(height / 32) * 32
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.BICUBIC)


def slice_clips(
    segment: pydub.AudioSegment,
    clip_duration_ms: int,
    overlap_ms: int,
) -> list[tuple[int, pydub.AudioSegment]]:
    increment = clip_duration_ms - overlap_ms
    starts = list(range(0, max(1, len(segment) - clip_duration_ms + 1), increment))
    if not starts:
        starts = [0]
    clips = []
    for start in starts:
        end = min(start + clip_duration_ms, len(segment))
        clip = segment[start:end]
        if len(clip) < clip_duration_ms:
            clip = clip + pydub.AudioSegment.silent(
                duration=clip_duration_ms - len(clip), frame_rate=clip.frame_rate
            )
        clips.append((start, clip))
    return clips


def stitch_clips(
    clips: list[pydub.AudioSegment],
    overlap_ms: int,
) -> pydub.AudioSegment:
    if len(clips) == 1:
        return clips[0]
    return clips[0].append(clips[1:], crossfade=overlap_ms)


def load_img2img(checkpoint: str, device: str, scheduler: str) -> StableDiffusionImg2ImgPipeline:
    dtype = torch.float32 if device in {"cpu", "mps", "xpu"} else torch.float16
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=dtype,
        safety_checker=lambda images, **kwargs: (images, False),
    ).to(device)
    pipeline.scheduler = get_scheduler(scheduler, config=pipeline.scheduler.config)
    return pipeline


def style_transfer_segment(
    segment: pydub.AudioSegment,
    pipeline: StableDiffusionImg2ImgPipeline,
    converter: SpectrogramImageConverter,
    prompt: str,
    negative_prompt: str,
    denoising: float,
    guidance: float,
    steps: int,
    seed: int,
    device: str,
    clip_duration_s: float,
    overlap_s: float,
) -> pydub.AudioSegment:
    clip_duration_ms = int(clip_duration_s * 1000)
    overlap_ms = int(overlap_s * 1000)
    clips = slice_clips(segment, clip_duration_ms, overlap_ms)
    styled: list[pydub.AudioSegment] = []

    generator_device = "cpu" if device.startswith("mps") else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    for i, (_start, clip) in enumerate(clips):
        print(f"  style clip {i + 1}/{len(clips)}")
        init_image = converter.spectrogram_image_from_audio(clip)
        init_resized = scale_image_to_32_stride(init_image)

        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=init_resized,
            strength=denoising,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        )
        image = output.images[0].resize(init_image.size, Image.BICUBIC)
        styled.append(converter.audio_from_spectrogram_image(image))

    return stitch_clips(styled, overlap_ms)[: len(segment)]


def match_length(segment: pydub.AudioSegment, target_ms: int) -> pydub.AudioSegment:
    if len(segment) > target_ms:
        return segment[:target_ms]
    if len(segment) < target_ms:
        return segment + pydub.AudioSegment.silent(
            duration=target_ms - len(segment), frame_rate=segment.frame_rate
        )
    return segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Folk-metal style convert with vocal lock")
    parser.add_argument("--input", required=True, type=Path, help="Source audio path")
    parser.add_argument("--output", type=Path, default=None, help="Output wav/mp3 path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--denoising", type=float, default=0.42)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--demucs-device", default="cpu")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scheduler", default="DPMSolverMultistepScheduler")
    parser.add_argument("--clip-duration", type=float, default=5.0)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument(
        "--keep-drums",
        action="store_true",
        default=True,
        help="Keep original drums (default on)",
    )
    parser.add_argument("--no-keep-drums", action="store_false", dest="keep_drums")
    parser.add_argument("--vocal-gain-db", type=float, default=0.0)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    out = args.output
    if out is None:
        out = args.input.with_name(f"{args.input.stem}_folk_metal.wav")

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Loading audio: {args.input}")
    original = load_audio(args.input)

    print("Splitting stems with Demucs (this can take a while)...")
    stems = split_audio(original, model_name="htdemucs", extension="wav", device=args.demucs_device)
    print(f"Stems: {sorted(stems)}")

    vocals = stems.get("vocals")
    drums = stems.get("drums")
    bass = stems.get("bass")
    other = stems.get("other")
    if vocals is None or other is None:
        raise SystemExit(f"Expected vocals/other stems, got: {list(stems)}")

    to_style = [other]
    if bass is not None and args.keep_drums:
        to_style.append(bass)
    elif bass is not None and not args.keep_drums:
        to_style.append(bass)
    if drums is not None and not args.keep_drums:
        to_style.append(drums)

    instrumental = overlay_sum(to_style)
    print(
        f"Style-transferring instrumental ({len(instrumental) / 1000:.1f}s), "
        f"denoising={args.denoising}, keep_drums={args.keep_drums}"
    )

    params = SpectrogramParams(min_frequency=0, max_frequency=10000, stereo=False)
    converter = SpectrogramImageConverter(params=params, device=device)
    pipeline = load_img2img(args.checkpoint, device=device, scheduler=args.scheduler)

    styled = style_transfer_segment(
        segment=instrumental,
        pipeline=pipeline,
        converter=converter,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        denoising=args.denoising,
        guidance=args.guidance,
        steps=args.steps,
        seed=args.seed,
        device=device,
        clip_duration_s=args.clip_duration,
        overlap_s=args.overlap,
    )

    target_ms = len(original)
    parts = [match_length(styled, target_ms), match_length(vocals + args.vocal_gain_db, target_ms)]
    if args.keep_drums and drums is not None:
        parts.append(match_length(drums, target_ms))

    final = overlay_sum(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lstrip(".") or "wav"
    final.export(out, format=fmt)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
