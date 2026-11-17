#!/usr/bin/env python3
"""Demo: generate a movie using the h3 CLI command via SSH.

Runs `h3 generate` on the AMD GPU host for each shot, downloads
the output clips, and assembles them into a demo movie with ffmpeg.

Usage:
    python demo_h3.py                                  # generate all shots
    python demo_h3.py --host 192.168.1.123             # custom SSH host
    python demo_h3.py --preset draft                   # fast preview
    python demo_h3.py --shots 1                        # generate only shot 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

DEMO_SHOTS = [
    {
        "id": "shot1",
        "prompt": (
            "Authentic early-1990s Japanese cel anime: hand-inked lines, "
            "two-tone cel shading, painted café background, soft film grain. "
            "Medium shot: MIMI, a cat-eared café waitress with a brown bob, "
            "green eyes, a red bow and a black-and-white uniform, proudly "
            "finishes a towering parfait, adds a cherry on top and lifts the "
            "glass onto her tray, tail swishing. "
            "SOUND: café chatter, clinking spoons, a bouncy 90s city-pop tune."
        ),
    },
    {
        "id": "shot2",
        "prompt": (
            "Authentic early-1990s Japanese cel anime: hand-inked lines, "
            "two-tone cel shading, painted café background, soft film grain. "
            "Hard cut, close-up: the cherry slides off the top; MIMI spins, "
            "catches it in her mouth mid-air, and winks at camera with the "
            "cherry stem in her teeth. "
            "SOUND: a boing on the catch, a bouncy 90s city-pop tune. "
            "Nobody speaks. No subtitles, no captions, no text."
        ),
    },
]


def h3_generate(host: str, shot: dict, preset: str, seed: int) -> str:
    """Run `h3 generate` on the remote host via SSH, return remote output path."""
    work = f"arcreel_{shot['id']}"
    remote_output = f"/data/h3/runs/{work}/output.mp4"

    print(f"\n{'='*60}")
    print(f"  Generating {shot['id']} via h3 CLI on {host}")
    print(f"  preset={preset}  seed={seed}  work={work}")
    print(f"{'='*60}")

    import shlex
    remote_cmd = (
        f"H3_SNAPSHOT=/data/h3/bf16/snapshot "
        f"HF_HOME=/data/h3/hf "
        f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 "
        f"~/.local/bin/h3 generate {preset} "
        f"--work {shlex.quote(work)} "
        f"--seed {seed} "
        f"-p {shlex.quote(shot['prompt'])}"
    )
    cmd = ["ssh", host, remote_cmd]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=3600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        sys.exit(1)

    print(f"  Done in {elapsed:.0f}s")
    return remote_output


def download_clip(host: str, remote_path: str, local_path: Path) -> None:
    """Download a clip from the remote host via scp."""
    subprocess.run(
        ["scp", f"{host}:{remote_path}", str(local_path)],
        check=True,
    )
    print(f"  Downloaded: {local_path} ({local_path.stat().st_size / 1024:.0f} KiB)")


def concat_clips(paths: list[Path], output: Path) -> None:
    """Concatenate mp4 clips using ffmpeg."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{p.resolve()}'\n")
        concat_file = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_file, "-c", "copy", str(output)],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        print("ffmpeg not found — using last clip as output")
        import shutil
        shutil.copy2(paths[-1], output)
    finally:
        Path(concat_file).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo movie via h3 CLI")
    parser.add_argument("--host", default="192.168.1.123",
                        help="SSH host with h3 and GPU (default: 192.168.1.123)")
    parser.add_argument("--output", default="demo_movie.mp4")
    parser.add_argument("--preset", default="balanced",
                        choices=["draft", "fast", "balanced", "high", "best"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=0,
                        help="Generate only the first N shots (0=all)")
    args = parser.parse_args()

    shots = DEMO_SHOTS[:args.shots] if args.shots > 0 else DEMO_SHOTS

    print("ArcReel Demo Movie Generator (h3 CLI mode)")
    print(f"  Host: {args.host}")
    print(f"  Preset: {args.preset}")
    print(f"  Shots: {len(shots)}")
    print(f"  Output: {args.output}")

    # Verify h3 is reachable
    check = subprocess.run(
        ["ssh", args.host, "~/.local/bin/h3 --version"],
        capture_output=True, text=True, timeout=10,
    )
    if check.returncode != 0:
        print(f"  Cannot reach h3 on {args.host}")
        sys.exit(1)
    print(f"  h3 version: {check.stdout.strip()}")

    clip_dir = Path("demo_clips")
    clip_dir.mkdir(exist_ok=True)

    clip_paths: list[Path] = []
    total_start = time.time()

    for i, shot in enumerate(shots):
        remote_path = h3_generate(args.host, shot, args.preset, args.seed + i)
        local_clip = clip_dir / f"{shot['id']}.mp4"
        download_clip(args.host, remote_path, local_clip)
        clip_paths.append(local_clip)

    # Assemble
    output = Path(args.output)
    if len(clip_paths) > 1:
        print(f"\nAssembling {len(clip_paths)} clips...")
        concat_clips(clip_paths, output)
    elif clip_paths:
        import shutil
        shutil.copy2(clip_paths[0], output)

    total_time = time.time() - total_start
    print(f"\nDemo movie: {output}")
    print(f"Size: {output.stat().st_size / 1024:.0f} KiB")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("NOTE: human must watch for quality verification")


if __name__ == "__main__":
    main()
