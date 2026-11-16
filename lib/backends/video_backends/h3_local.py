"""H3 local GPU backend — runs MiniMax H3 via the ``h3`` CLI on a local or SSH-reachable AMD GPU.

This backend calls the ``h3`` binary (https://github.com/yijunyu/h3) which orchestrates
the Python pipeline (h3amd.py) with temporal-window attention for 4x speedup on gfx1151.

Unlike cloud backends, this one is synchronous from the caller's perspective: ``generate``
blocks until the video file is on disk. There is no provider job ID or resume capability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arcreel_catalog_core import CatalogEntry
from arcreel_market_core.video_backend_contract import (
    VideoAudioMode,
    VideoCapabilities,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from lib.backends.video_backends.name_family import catalog_entry_of
from lib.billing.metering import Usage

logger = logging.getLogger(__name__)

# Quality presets map to window sizes.
_PRESETS: dict[str, int] = {
    "draft": 6,
    "fast": 8,
    "balanced": 12,
    "high": 18,
    "best": 0,
}

# Aspect ratio -> (width, height) for supported resolutions.
_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
}
_DEFAULT_SIZE = (1344, 768)


@dataclass
class H3LocalConfig:
    """Configuration for the local H3 backend."""

    h3_binary: str = "h3"
    ssh_host: str | None = None
    preset: str = "balanced"
    seed: int = 42
    h3_root: str = "/data/h3"


class H3LocalVideoBackend:
    """Video backend that generates clips on a local AMD GPU via the ``h3`` CLI."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        model: str = "minimax-h3-local",
        config: H3LocalConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or H3LocalConfig()
        self._window = _PRESETS.get(self._config.preset, 12)

    @property
    def name(self) -> str:
        return "h3-local"

    @property
    def model(self) -> str:
        return self._model

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return VideoCapabilities(
            text_to_video=True,
            first_frame=True,
            last_frame=False,
            max_reference_images=0,
            audio_track=VideoAudioMode.ALWAYS_ON,
        )

    @staticmethod
    def video_capabilities_for_model(
        model_id: str, entry: CatalogEntry | None = None
    ) -> VideoCapabilities:
        return VideoCapabilities(
            text_to_video=True,
            first_frame=True,
            last_frame=False,
            max_reference_images=0,
            audio_track=VideoAudioMode.ALWAYS_ON,
        )

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        cfg = self._config
        width, height = _ASPECT_SIZES.get(request.aspect_ratio, _DEFAULT_SIZE)

        work_name = f"arcreel_{request.task_id or 'gen'}"
        prompt_text = request.prompt

        cmd: list[str] = []

        if cfg.ssh_host:
            cmd = ["ssh", cfg.ssh_host]

        cmd += [
            cfg.h3_binary,
            "--window", str(self._window),
            "--work", work_name,
            "-p", prompt_text,
            "-s", str(request.duration_seconds),
            "-W", str(width),
            "-H", str(height),
            "--seed", str(cfg.seed),
        ]

        if request.start_image:
            cmd += ["--image", str(request.start_image)]

        logger.info("h3-local: running %s", " ".join(cmd[:6]) + " ...")

        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            ),
        )

        if proc.returncode != 0:
            logger.error("h3-local failed (exit %d): %s", proc.returncode, proc.stderr[-500:] if proc.stderr else "")
            raise RuntimeError(f"h3 generation failed (exit {proc.returncode})")

        remote_output = Path(cfg.h3_root) / "runs" / work_name / "output.mp4"

        if cfg.ssh_host:
            scp_proc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["scp", f"{cfg.ssh_host}:{remote_output}", str(request.output_path)],
                    capture_output=True,
                    timeout=120,
                ),
            )
            if scp_proc.returncode != 0:
                raise RuntimeError(f"scp failed: {scp_proc.stderr}")
        else:
            shutil.copy2(remote_output, request.output_path)

        return VideoGenerationResult(
            video_path=request.output_path,
            provider="h3-local",
            model=self._model,
            duration_seconds=request.duration_seconds,
            seed=cfg.seed,
        )

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        raise NotImplementedError("h3-local does not support resume — generation is synchronous")
