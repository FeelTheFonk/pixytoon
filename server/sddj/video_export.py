"""MP4 video export — combines animation frames + audio via ffmpeg."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("sddj.video_export")

# Quality presets: (crf, preset, default_scale)
QUALITY_PRESETS: dict[str, tuple[int, str, int]] = {
    "web":     (23, "medium",   4),
    "high":    (17, "slow",     4),
    "archive": (12, "veryslow", 8),
    "raw":     (0,  "ultrafast", 1),
}

_SAFE_METADATA_RE = re.compile(r"[^\w\s.,;:!?@#$%^&*()\-+=\[\]{}<>/\\|~`'\"]")

_FRAME_NUM_RE = re.compile(r"frame_(\d+)\.png$")


def find_ffmpeg() -> str | None:
    """Return the path to ffmpeg if available, else None."""
    return shutil.which("ffmpeg")


def _fill_frame_gaps(frame_dir: Path, frames: list[Path]) -> list[Path]:
    """Ensure sequential frame numbering with no gaps.

    If frame_003.png is missing between frame_002.png and frame_004.png,
    copy frame_002.png → frame_003.png (forward fill).  ffmpeg's image2
    demuxer stops at the first gap, so this is critical for complete export.

    Returns the updated sorted frame list.
    """
    # Parse frame numbers
    numbered: dict[int, Path] = {}
    for f in frames:
        m = _FRAME_NUM_RE.search(f.name)
        if m:
            numbered[int(m.group(1))] = f

    if not numbered:
        return frames

    first = min(numbered)
    last = max(numbered)
    filled = 0
    prev_path = numbered[first]

    for n in range(first, last + 1):
        if n in numbered:
            prev_path = numbered[n]
        else:
            # Gap detected — forward-fill with previous frame
            gap_path = frame_dir / f"frame_{n:03d}.png"
            shutil.copy2(prev_path, gap_path)
            numbered[n] = gap_path
            filled += 1

    if filled:
        log.warning("Filled %d gap(s) in frame sequence [%d..%d]", filled, first, last)

    return sorted(numbered.values())


@dataclass
class ExportResult:
    path: str
    size_mb: float
    duration_s: float


def export_mp4(
    frame_dir: str | Path,
    audio_path: str | Path | None,
    fps: float = 24.0,
    scale_factor: int = 4,
    quality: str = "high",
    metadata: dict[str, str] | None = None,
    ffmpeg_path: str | None = None,
) -> ExportResult:
    """Export PNG frames + audio to MP4 using ffmpeg.

    Args:
        frame_dir: Directory containing frame_001.png, frame_002.png, ...
        audio_path: Path to the audio file (optional, can be None for silent video).
        fps: Frame rate.
        scale_factor: Nearest-neighbor upscale factor (1 = no scaling).
        quality: One of "web", "high", "archive", "raw".
        metadata: Optional dict of metadata to embed (e.g. prompt, seed).
        ffmpeg_path: Path to ffmpeg binary (auto-detected if None).

    Returns:
        ExportResult with output path and file size.

    Raises:
        FileNotFoundError: If ffmpeg is not found or frame_dir doesn't exist.
        ValueError: If no frames found or invalid quality preset.
        RuntimeError: If ffmpeg fails.
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    # Validate frames exist
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise ValueError(f"No frame_*.png files found in {frame_dir}")

    # Fill any gaps in the numbering sequence (critical — ffmpeg stops at gaps)
    frames = _fill_frame_gaps(frame_dir, frames)

    # Detect start number from first frame
    m = _FRAME_NUM_RE.search(frames[0].name)
    start_number = int(m.group(1)) if m else 1

    frame_pattern = str(frame_dir / "frame_%03d.png")

    # Find ffmpeg
    ffmpeg = ffmpeg_path or find_ffmpeg()
    if ffmpeg is None:
        raise FileNotFoundError(
            "ffmpeg not found in PATH. Install ffmpeg or set SDDJ_FFMPEG_PATH."
        )

    # Quality preset
    if quality not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality preset: {quality}. Use: {list(QUALITY_PRESETS)}")
    crf, preset, default_scale = QUALITY_PRESETS[quality]
    if scale_factor <= 0:
        scale_factor = default_scale

    # Validate audio path if provided
    if audio_path is not None:
        audio_path = Path(audio_path)
        real_audio = Path(os.path.realpath(audio_path))
        if not real_audio.is_file():
            log.warning("Audio file not found: %s — exporting without audio", audio_path)
            audio_path = None
        else:
            audio_path = real_audio

    # Output path
    output_path = frame_dir / "video.mp4"

    # Build ffmpeg command
    cmd = [
        ffmpeg,
        "-y",  # overwrite
        "-start_number", str(start_number),
        "-framerate", str(fps),
        "-i", frame_pattern,
    ]

    # Add audio input
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])

    # Video filter chain:
    # 1. colorchannelmixer=aa=1 — force alpha to opaque (transparent frames → black)
    # 2. scale with nearest-neighbor — crisp upscaling
    vf_parts = ["colorchannelmixer=aa=1"]
    if scale_factor > 1:
        vf_parts.append(f"scale=iw*{scale_factor}:ih*{scale_factor}:flags=neighbor")
    cmd.extend(["-vf", ",".join(vf_parts)])

    # Video codec
    if quality == "raw":
        cmd.extend(["-c:v", "libx264", "-qp", "0"])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
        ])

    # Pixel format for compatibility
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Audio codec
    if audio_path is not None:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
        ])

    # Fast start for web streaming
    cmd.extend(["-movflags", "+faststart"])

    # Metadata
    if metadata:
        for key, value in metadata.items():
            safe_value = _SAFE_METADATA_RE.sub("", str(value))[:256]
            if safe_value:
                cmd.extend(["-metadata", f"{key}={safe_value}"])

    cmd.append(str(output_path))

    log.info("Exporting MP4: %d frames, fps=%.1f, scale=%dx, quality=%s",
             len(frames), fps, scale_factor, quality)
    log.debug("ffmpeg command: %s", " ".join(cmd))

    # Run ffmpeg
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out after 300 seconds")

    if result.returncode != 0:
        stderr = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {stderr}")

    if not output_path.is_file():
        raise RuntimeError("ffmpeg completed but output file not found")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    duration_s = len(frames) / fps

    log.info("MP4 exported: %s (%.1f MB, %.1fs)", output_path, size_mb, duration_s)

    return ExportResult(
        path=str(output_path),
        size_mb=round(size_mb, 2),
        duration_s=round(duration_s, 2),
    )
