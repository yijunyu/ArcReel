#!/usr/bin/env python3
"""Demo: generate a movie through the h3 serve API.

Uses the same MiniMax H3 submit/poll protocol that ArcReel's
declarative endpoint uses, proving the integration path works.

Usage:
    python demo_h3.py                              # default: http://192.168.1.123:8321
    python demo_h3.py --api http://localhost:8321   # local
    python demo_h3.py --reuse-existing              # skip generation, use existing clips
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
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


def api_post(base: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_get(base: str, path: str) -> dict:
    req = urllib.request.Request(f"{base}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=600) as resp:
        dest.write_bytes(resp.read())


def submit_and_poll(base: str, shot: dict) -> str:
    """Submit a shot via the MiniMax H3 API and poll until done."""
    print(f"\n{'='*60}")
    print(f"  Submitting {shot['id']}")
    print(f"{'='*60}")

    body = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": shot["prompt"]}],
        "resolution": "768P",
        "duration": 5,
    }

    result = api_post(base, "/v2/video_generation", body)
    task_id = result["task_id"]
    print(f"  task_id: {task_id}")

    start = time.time()
    while True:
        time.sleep(30)
        elapsed = time.time() - start
        try:
            status = api_get(base, f"/v2/query/video_generation/{task_id}")
        except Exception as e:
            print(f"  [{elapsed:.0f}s] poll error: {e}")
            continue

        task = status.get("task", {})
        st = task.get("status", "unknown")
        print(f"  [{elapsed:.0f}s] status: {st}")

        if st == "succeeded":
            video_url = task.get("content", {}).get("url", "")
            print(f"  video: {video_url}")
            return video_url
        elif st == "failed":
            err = task.get("error", "unknown")
            print(f"  FAILED: {err}")
            sys.exit(1)
        elif elapsed > 3600:
            print("  TIMEOUT after 1h")
            sys.exit(1)


def concat_local(paths: list[Path], output: Path) -> None:
    """Concatenate mp4 files using ffmpeg."""
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
    parser = argparse.ArgumentParser(description="Demo movie via h3 serve API")
    parser.add_argument("--api", default="http://192.168.1.123:8321",
                        help="h3 serve base URL")
    parser.add_argument("--output", default="demo_movie.mp4")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Use existing clips from AMD host instead of generating")
    parser.add_argument("--ssh-host", default="amd",
                        help="SSH host for scp (when --reuse-existing)")
    args = parser.parse_args()

    print("H3 Demo Movie Generator (API mode)")
    print(f"  API: {args.api}")
    print(f"  Shots: {len(DEMO_SHOTS)}")
    print(f"  Output: {args.output}")

    # Check API health
    try:
        health = api_get(args.api, "/health")
        print(f"  Health: {health}")
    except Exception as e:
        print(f"  Cannot reach API at {args.api}: {e}")
        print("  Start h3 serve on the AMD host first:")
        print("    ssh amd 'nohup ~/.local/bin/h3 serve > /tmp/h3serve.log 2>&1 &'")
        sys.exit(1)

    clip_dir = Path("demo_clips")
    clip_dir.mkdir(exist_ok=True)

    clip_paths: list[Path] = []

    if args.reuse_existing:
        # Reuse existing win12 output as both shots (same prompt)
        existing = ["runs/win12/output.mp4", "runs/recbf/output.mp4"]
        for i, shot in enumerate(DEMO_SHOTS):
            clip = clip_dir / f"{shot['id']}.mp4"
            src = existing[i % len(existing)]
            subprocess.run(
                ["scp", f"{args.ssh_host}:/data/h3/{src}", str(clip)],
                check=True,
            )
            clip_paths.append(clip)
            print(f"  Reused {src} as {shot['id']}")
    else:
        # Generate each shot through the API
        for shot in DEMO_SHOTS:
            video_url = submit_and_poll(args.api, shot)
            clip = clip_dir / f"{shot['id']}.mp4"
            print(f"  Downloading {clip.name}...")
            download(video_url, clip)
            clip_paths.append(clip)
            print(f"  Saved: {clip}")

    # Concatenate
    output = Path(args.output)
    if len(clip_paths) > 1:
        concat_local(clip_paths, output)
    elif clip_paths:
        import shutil
        shutil.copy2(clip_paths[0], output)

    print(f"\nDemo movie: {output}")
    print(f"Size: {output.stat().st_size / 1024:.0f} KiB")
    print("NOTE: human must watch for quality verification")


if __name__ == "__main__":
    main()
