"""H3LocalVideoBackend: protocol compliance and generation path through ArcReel's VideoBackend contract."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from arcreel_market_core.video_backend_contract import (
    ReferenceAudioMode,
    VideoAudioMode,
    VideoGenerationRequest,
    VideoGenerationResult,
)
from lib.backends.video_backends.h3_local import H3LocalVideoBackend


def test_capabilities_match_h3_workflows():
    backend = H3LocalVideoBackend()
    caps = backend.video_capabilities
    assert caps.text_to_video is True
    assert caps.first_frame is True
    assert caps.last_frame is False
    assert caps.max_reference_images == 1
    assert caps.audio_track == VideoAudioMode.ALWAYS_ON
    assert caps.reference_audio_mode == ReferenceAudioMode.NONE
    assert caps.max_reference_audio_count == 0


def test_protocol_properties():
    backend = H3LocalVideoBackend()
    assert backend.name == "h3-local"
    assert backend.model == "minimax-h3-flex"
    assert backend.video_capabilities is not None


async def test_resume_video_raises():
    backend = H3LocalVideoBackend()
    import pytest

    with pytest.raises(NotImplementedError):
        await backend.resume_video("job-1", VideoGenerationRequest(prompt="x", output_path=Path("/tmp/x.mp4")))


def _make_fake_exec(returncode: int = 0, stdout: bytes = b"done"):
    """Create a fake asyncio.create_subprocess_exec that returns a mock process."""

    async def fake_exec(*cmd, **_kw):
        proc = AsyncMock()
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, None))
        return proc

    return fake_exec


async def _patched_wait_for(coro, **_kw):
    return await coro


async def test_generate_builds_correct_ssh_command(tmp_path: Path):
    output = tmp_path / "out.mp4"
    output.write_bytes(b"\x00" * 1024)

    backend = H3LocalVideoBackend(host="testhost", preset="draft")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_make_fake_exec()) as mock_exec,
        patch("asyncio.wait_for", side_effect=_patched_wait_for),
    ):
        result = await backend.generate(
            VideoGenerationRequest(
                prompt="A cat walking",
                output_path=output,
                seed=42,
            )
        )

    assert isinstance(result, VideoGenerationResult)
    assert result.provider == "h3-local"
    assert result.model == "minimax-h3-flex"

    calls = mock_exec.call_args_list
    assert len(calls) == 2
    ssh_call = calls[0]
    ssh_args = ssh_call[0]
    assert ssh_args[0] == "ssh"
    assert ssh_args[1] == "testhost"
    remote_cmd = ssh_args[2]
    assert "h3 generate draft" in remote_cmd
    assert "A cat walking" in remote_cmd or "'A cat walking'" in remote_cmd
    assert "--seed" in remote_cmd
    assert "42" in remote_cmd

    scp_call = calls[1]
    scp_args = scp_call[0]
    assert scp_args[0] == "scp"
    assert "testhost:" in scp_args[1]


async def test_generate_with_start_image(tmp_path: Path):
    output = tmp_path / "out.mp4"
    output.write_bytes(b"\x00" * 1024)
    start_img = tmp_path / "frame.png"
    start_img.write_bytes(b"\x89PNG")

    backend = H3LocalVideoBackend(host="testhost")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_make_fake_exec()) as mock_exec,
        patch("asyncio.wait_for", side_effect=_patched_wait_for),
    ):
        await backend.generate(
            VideoGenerationRequest(
                prompt="Animate this",
                output_path=output,
                start_image=start_img,
            )
        )

    ssh_cmd = mock_exec.call_args_list[0][0][2]
    assert "--image" in ssh_cmd


async def test_generate_with_reference_images(tmp_path: Path):
    output = tmp_path / "out.mp4"
    output.write_bytes(b"\x00" * 1024)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG")

    backend = H3LocalVideoBackend(host="testhost")

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_make_fake_exec()) as mock_exec,
        patch("asyncio.wait_for", side_effect=_patched_wait_for),
    ):
        await backend.generate(
            VideoGenerationRequest(
                prompt="Reference style",
                output_path=output,
                reference_images=[ref],
            )
        )

    ssh_cmd = mock_exec.call_args_list[0][0][2]
    assert "--ref" in ssh_cmd


async def test_generate_failure_raises(tmp_path: Path):
    output = tmp_path / "out.mp4"
    backend = H3LocalVideoBackend(host="testhost")
    import pytest

    with (
        patch("asyncio.create_subprocess_exec", side_effect=_make_fake_exec(returncode=1, stdout=b"GPU error")),
        patch("asyncio.wait_for", side_effect=_patched_wait_for),
        pytest.raises(RuntimeError, match="h3 generate failed"),
    ):
        await backend.generate(VideoGenerationRequest(prompt="fail", output_path=output))


def test_create_backend_integration():
    from lib.backends.video_backends import create_backend

    backend = create_backend("h3-local")
    assert backend.name == "h3-local"
    assert backend.model == "minimax-h3-flex"
    caps = backend.video_capabilities
    assert caps.text_to_video is True
    assert caps.first_frame is True
    assert caps.audio_track == VideoAudioMode.ALWAYS_ON
