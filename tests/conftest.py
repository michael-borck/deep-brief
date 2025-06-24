"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Generator
import tempfile
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_video_path() -> Path:
    """Path to a sample video file for testing."""
    # For now, return a mock path - we'll create actual test videos later
    return Path("tests/fixtures/sample_video.mp4")


@pytest.fixture
def sample_audio_path() -> Path:
    """Path to a sample audio file for testing."""
    return Path("tests/fixtures/sample_audio.wav")


@pytest.fixture
def config_dict() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "processing": {
            "max_video_size_mb": 500,
            "supported_formats": ["mp4", "mov", "avi", "webm"],
        },
        "scene_detection": {
            "method": "threshold",
            "threshold": 0.4,
            "min_scene_duration": 2.0,
            "fallback_interval": 30.0,
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
        },
        "transcription": {
            "model": "whisper-base",
            "language": "auto",
            "word_timestamps": True,
        },
        "analysis": {
            "filler_words": ["um", "uh", "like", "you know", "so"],
            "target_wpm_range": [140, 160],
        },
        "output": {
            "formats": ["json", "html"],
            "include_frames": True,
            "frame_quality": 80,
        },
    }