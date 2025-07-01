"""Utilities for video testing."""

import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest


class VideoInfo(NamedTuple):
    """Video file information."""

    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    file_size: int
    format: str


def get_video_info(video_path: Path) -> VideoInfo | None:
    """
    Get basic information about a video file using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object or None if ffprobe fails
    """
    if not video_path.exists():
        return None

    try:
        # Use ffprobe to get video information
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        import json

        data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        audio_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
            elif stream.get("codec_type") == "audio":
                audio_stream = stream

        if not video_stream:
            return None

        # Extract format info
        format_info = data.get("format", {})

        return VideoInfo(
            duration=float(format_info.get("duration", 0)),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=eval(video_stream.get("r_frame_rate", "0/1")),  # e.g., "30/1"
            has_audio=audio_stream is not None,
            file_size=int(format_info.get("size", 0)),
            format=format_info.get("format_name", "unknown"),
        )

    except (subprocess.CalledProcessError, json.JSONDecodeError, Exception):
        return None


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_ffprobe_available() -> bool:
    """Check if ffprobe is available on the system."""
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def skip_if_no_ffmpeg():
    """Pytest decorator to skip tests if ffmpeg is not available."""
    return pytest.mark.skipif(not is_ffmpeg_available(), reason="ffmpeg not available")


def skip_if_no_ffprobe():
    """Pytest decorator to skip tests if ffprobe is not available."""
    return pytest.mark.skipif(
        not is_ffprobe_available(), reason="ffprobe not available"
    )


def extract_audio_from_video(video_path: Path, output_path: Path) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Source video file
        output_path: Output audio file path

    Returns:
        True if successful, False otherwise
    """
    if not is_ffmpeg_available():
        return False

    try:
        cmd = [
            "ffmpeg",
            "-y",  # -y to overwrite
            "-i",
            str(video_path),
            "-vn",  # no video
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # mono
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path.exists()

    except subprocess.CalledProcessError:
        return False


def create_test_audio(
    output_path: Path, duration: int = 5, frequency: int = 440
) -> bool:
    """
    Create a test audio file using ffmpeg.

    Args:
        output_path: Output audio file path
        duration: Duration in seconds
        frequency: Audio frequency in Hz

    Returns:
        True if successful, False otherwise
    """
    if not is_ffmpeg_available():
        return False

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "16000",
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path.exists()

    except subprocess.CalledProcessError:
        return False


def assert_valid_video(video_path: Path, min_duration: float = 0.1) -> None:
    """
    Assert that a video file is valid and meets basic requirements.

    Args:
        video_path: Path to video file
        min_duration: Minimum expected duration

    Raises:
        AssertionError: If video is invalid
    """
    assert video_path.exists(), f"Video file does not exist: {video_path}"
    assert video_path.stat().st_size > 0, f"Video file is empty: {video_path}"

    info = get_video_info(video_path)
    assert info is not None, f"Could not read video info: {video_path}"
    assert info.duration >= min_duration, (
        f"Video too short: {info.duration}s < {min_duration}s"
    )
    assert info.width > 0 and info.height > 0, (
        f"Invalid video dimensions: {info.width}x{info.height}"
    )


def assert_video_has_audio(video_path: Path) -> None:
    """
    Assert that a video file has an audio track.

    Args:
        video_path: Path to video file

    Raises:
        AssertionError: If video has no audio
    """
    info = get_video_info(video_path)
    assert info is not None, f"Could not read video info: {video_path}"
    assert info.has_audio, f"Video has no audio track: {video_path}"


def assert_video_no_audio(video_path: Path) -> None:
    """
    Assert that a video file has no audio track.

    Args:
        video_path: Path to video file

    Raises:
        AssertionError: If video has audio
    """
    info = get_video_info(video_path)
    assert info is not None, f"Could not read video info: {video_path}"
    assert not info.has_audio, f"Video has audio track when it shouldn't: {video_path}"


class VideoTestCase:
    """Base class for video testing with common utilities."""

    @pytest.fixture(autouse=True)
    def setup_video_test(self, temp_dir):
        """Set up temporary directory for video tests."""
        self.temp_dir = temp_dir
        self.output_dir = temp_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

    def get_output_path(self, filename: str) -> Path:
        """Get path for test output file."""
        return self.output_dir / filename

    def assert_output_exists(self, filename: str) -> Path:
        """Assert that an output file exists and return its path."""
        output_path = self.get_output_path(filename)
        assert output_path.exists(), f"Output file not found: {output_path}"
        return output_path
