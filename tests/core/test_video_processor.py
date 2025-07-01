"""Tests for video processor functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import ffmpeg
import pytest

from deep_brief.core.video_processor import VideoInfo, VideoProcessor
from deep_brief.utils.config import DeepBriefConfig, ProcessingConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = DeepBriefConfig(
        processing=ProcessingConfig(
            max_video_size_mb=100,
            supported_formats=["mp4", "mov", "avi", "webm"],
            temp_dir=Path(tempfile.mkdtemp()),
            cleanup_temp_files=True,
        )
    )
    return config


@pytest.fixture
def video_processor(mock_config):
    """Create VideoProcessor instance for testing."""
    return VideoProcessor(config=mock_config)


@pytest.fixture
def mock_probe_data():
    """Mock ffmpeg probe data."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            }
        ],
        "format": {"duration": "120.5", "format_name": "mp4"},
    }


class TestVideoProcessor:
    """Test VideoProcessor class."""

    def test_init_creates_temp_directory(self, mock_config):
        """Test that VideoProcessor creates temp directory on init."""
        processor = VideoProcessor(config=mock_config)
        assert processor.temp_dir.exists()
        assert processor.supported_formats == ["mp4", "mov", "avi", "webm"]

    def test_supported_formats_are_lowercase(self, mock_config):
        """Test that supported formats are converted to lowercase."""
        mock_config.processing.supported_formats = ["MP4", "MOV", "AVI"]
        processor = VideoProcessor(config=mock_config)
        assert processor.supported_formats == ["mp4", "mov", "avi"]

    def test_is_format_supported(self, video_processor):
        """Test format support checking."""
        assert video_processor.is_format_supported("test.mp4") is True
        assert video_processor.is_format_supported("test.MP4") is True
        assert video_processor.is_format_supported("test.mkv") is False
        assert video_processor.is_format_supported(Path("test.webm")) is True

    def test_get_supported_formats(self, video_processor):
        """Test getting supported formats list."""
        formats = video_processor.get_supported_formats()
        assert formats == ["mp4", "mov", "avi", "webm"]
        # Ensure it returns a copy
        formats.append("test")
        assert "test" not in video_processor.get_supported_formats()


class TestVideoValidation:
    """Test video file validation."""

    def test_validate_file_not_found(self, video_processor):
        """Test validation of non-existent file."""
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            video_processor.validate_file("nonexistent.mp4")

    def test_validate_unsupported_format(self, video_processor, tmp_path):
        """Test validation of unsupported format."""
        # Create a dummy file with unsupported extension
        test_file = tmp_path / "test.mkv"
        test_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported format 'mkv'"):
            video_processor.validate_file(test_file)

    def test_validate_file_too_large(self, video_processor, tmp_path):
        """Test validation of oversized file."""
        # Create a file larger than the configured limit
        test_file = tmp_path / "large.mp4"
        # Write more than 100MB (config limit)
        large_content = "x" * (101 * 1024 * 1024)
        test_file.write_text(large_content)

        with pytest.raises(ValueError, match="File too large"):
            video_processor.validate_file(test_file)

    @patch("ffmpeg.probe")
    def test_validate_file_success(
        self, mock_probe, video_processor, tmp_path, mock_probe_data
    ):
        """Test successful file validation."""
        # Create a small test file
        test_file = tmp_path / "test.mp4"
        test_file.write_text("small video content")

        # Mock ffmpeg.probe response
        mock_probe.return_value = mock_probe_data

        result = video_processor.validate_file(test_file)

        assert isinstance(result, VideoInfo)
        assert result.file_path == test_file
        assert result.duration == 120.5
        assert result.width == 1920
        assert result.height == 1080
        assert result.fps == 30.0
        assert result.format == "mp4"
        assert result.codec == "h264"

    @patch("ffmpeg.probe")
    def test_validate_file_no_video_stream(self, mock_probe, video_processor, tmp_path):
        """Test validation of file without video stream."""
        test_file = tmp_path / "audio_only.mp4"
        test_file.write_text("audio content")

        # Mock probe data with no video stream
        mock_probe.return_value = {
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "60.0", "format_name": "mp4"},
        }

        with pytest.raises(ValueError, match="No video stream found"):
            video_processor.validate_file(test_file)

    @patch("ffmpeg.probe")
    def test_validate_file_ffmpeg_error(self, mock_probe, video_processor, tmp_path):
        """Test handling of ffmpeg errors."""
        test_file = tmp_path / "corrupted.mp4"
        test_file.write_text("corrupted content")

        # Mock ffmpeg error
        mock_probe.side_effect = ffmpeg.Error("ffmpeg", "stdout", "stderr")

        with pytest.raises(RuntimeError, match="Cannot read video file"):
            video_processor.validate_file(test_file)

    @patch("ffmpeg.probe")
    def test_validate_file_complex_fps(self, mock_probe, video_processor, tmp_path):
        """Test FPS calculation with different formats."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("video content")

        # Test fractional FPS
        probe_data = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "24000/1001",  # 23.976 fps
                }
            ],
            "format": {"duration": "120.0", "format_name": "mp4"},
        }
        mock_probe.return_value = probe_data

        result = video_processor.validate_file(test_file)
        assert abs(result.fps - 23.976) < 0.001

    @patch("ffmpeg.probe")
    def test_validate_file_zero_denominator_fps(
        self, mock_probe, video_processor, tmp_path
    ):
        """Test FPS calculation with zero denominator."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("video content")

        probe_data = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/0",  # Invalid FPS
                }
            ],
            "format": {"duration": "120.0", "format_name": "mp4"},
        }
        mock_probe.return_value = probe_data

        result = video_processor.validate_file(test_file)
        assert result.fps == 0.0


class TestTempFileCleanup:
    """Test temporary file cleanup functionality."""

    def test_cleanup_temp_files_enabled(self, video_processor):
        """Test cleanup when enabled."""
        # Create some temp files
        temp_file1 = video_processor.temp_dir / "temp1.txt"
        temp_file2 = video_processor.temp_dir / "temp2.txt"
        temp_file1.write_text("temp content 1")
        temp_file2.write_text("temp content 2")

        assert temp_file1.exists()
        assert temp_file2.exists()

        video_processor.cleanup_temp_files()

        assert not temp_file1.exists()
        assert not temp_file2.exists()

    def test_cleanup_temp_files_disabled(self, mock_config):
        """Test cleanup when disabled."""
        mock_config.processing.cleanup_temp_files = False
        processor = VideoProcessor(config=mock_config)

        # Create temp file
        temp_file = processor.temp_dir / "temp.txt"
        temp_file.write_text("temp content")

        processor.cleanup_temp_files()

        # File should still exist
        assert temp_file.exists()

    def test_cleanup_nonexistent_temp_dir(self, video_processor):
        """Test cleanup when temp directory doesn't exist."""
        video_processor.temp_dir.rmdir()  # Remove temp directory

        # Should not raise an error
        video_processor.cleanup_temp_files()

    def test_cleanup_handles_permission_errors(self, video_processor, caplog):
        """Test cleanup handles permission errors gracefully."""
        temp_file = video_processor.temp_dir / "readonly.txt"
        temp_file.write_text("content")

        # Mock unlink to raise permission error
        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            video_processor.cleanup_temp_files()

        # Should log warning but not crash
        assert "Could not remove temp file" in caplog.text
