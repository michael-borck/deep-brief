"""Example test to verify pytest configuration."""

from pathlib import Path

import pytest


def test_pytest_working() -> None:
    """Test that pytest is properly configured."""
    assert True


def test_temp_dir_fixture(temp_dir: Path) -> None:
    """Test that temp_dir fixture works."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()

    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, World!")

    assert test_file.exists()
    assert test_file.read_text() == "Hello, World!"


def test_config_fixture(config_dict: dict) -> None:
    """Test that config fixture provides expected structure."""
    assert "processing" in config_dict
    assert "scene_detection" in config_dict
    assert "transcription" in config_dict

    assert config_dict["processing"]["max_video_size_mb"] == 500
    assert config_dict["scene_detection"]["threshold"] == 0.4
    assert config_dict["transcription"]["model"] == "whisper-base"


@pytest.mark.slow
def test_slow_operation() -> None:
    """Example of a test marked as slow."""
    # This would be a test that takes a long time
    # Can be skipped with: pytest -m "not slow"
    assert True


class TestExampleClass:
    """Example test class to verify class-based tests work."""

    def test_method_one(self) -> None:
        """Test method in a class."""
        assert 1 + 1 == 2

    def test_method_two(self, config_dict: dict) -> None:
        """Test method using a fixture."""
        assert isinstance(config_dict, dict)
