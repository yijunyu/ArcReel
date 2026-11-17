"""H3LocalVideoBackend — 本地 GPU h3 CLI 视频生成后端。

通过 SSH 在远程 GPU 主机上调用 ``h3 generate`` 命令，下载生成的视频文件。
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from arcreel_market_core.video_backend_contract import (
    ReferenceAudioMode,
    VideoAudioMode,
    VideoCapabilities,
    VideoGenerationRequest,
    VideoGenerationResult,
)

logger = logging.getLogger(__name__)

_PROVIDER = "h3-local"


@dataclass
class H3LocalVideoBackend:
    """通过 SSH 调用远程 h3 CLI 的视频生成后端。

    MiniMax H3 Flex 本地 GPU 推理，三条工作流：
    - t2va: 纯文生视频+音频
    - fl2va: 首帧图生视频+音频 (``--image``)
    - ref2va: 参考图生视频+音频 (``--ref``, 需 transformer_ref 权重)
    """

    host: str = "192.168.1.123"
    h3_bin: str = "~/.local/bin/h3"
    preset: str = "draft"
    env: dict[str, str] | None = None

    _CAPABILITIES: ClassVar[VideoCapabilities] = VideoCapabilities(
        text_to_video=True,
        first_frame=True,
        last_frame=False,
        max_reference_images=1,
        audio_track=VideoAudioMode.ALWAYS_ON,
        reference_audio_mode=ReferenceAudioMode.NONE,
        max_reference_audio_count=0,
    )

    @property
    def name(self) -> str:
        return _PROVIDER

    @property
    def model(self) -> str:
        return "minimax-h3-flex"

    @property
    def video_capabilities(self) -> VideoCapabilities:
        return self._CAPABILITIES

    async def generate(self, request: VideoGenerationRequest) -> VideoGenerationResult:
        work = f"arcreel_{request.task_id or 'gen'}"
        env_parts = {
            "H3_SNAPSHOT": "/data/h3/bf16/snapshot",
            "HF_HOME": "/data/h3/hf",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
            **(self.env or {}),
        }
        env_str = " ".join(f"{k}={v}" for k, v in env_parts.items())

        cmd_parts = [
            self.h3_bin, "generate", self.preset,
            "--work", shlex.quote(work),
            "-p", shlex.quote(request.prompt),
        ]
        if request.start_image:
            cmd_parts.extend(["--image", shlex.quote(str(request.start_image))])
        if request.reference_images:
            for ref in request.reference_images:
                cmd_parts.extend(["--ref", shlex.quote(str(ref))])
        if request.seed is not None:
            cmd_parts.extend(["--seed", str(request.seed)])

        remote_cmd = f"{env_str} {' '.join(cmd_parts)}"
        ssh_cmd = ["ssh", self.host, remote_cmd]

        logger.info("h3-local: %s", " ".join(ssh_cmd))
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"h3 generate failed (exit {proc.returncode}): {stdout.decode()[-500:]}")

        remote_output = f"/data/h3/runs/{work}/output.mp4"
        scp_proc = await asyncio.create_subprocess_exec(
            "scp", f"{self.host}:{remote_output}", str(request.output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await asyncio.wait_for(scp_proc.communicate(), timeout=120)
        if scp_proc.returncode != 0:
            raise RuntimeError(f"scp download failed for {remote_output}")

        return VideoGenerationResult(
            video_path=request.output_path,
            provider=_PROVIDER,
            model=self.model,
            duration_seconds=request.duration_seconds,
        )

    async def resume_video(self, job_id: str, request: VideoGenerationRequest) -> VideoGenerationResult:
        raise NotImplementedError
