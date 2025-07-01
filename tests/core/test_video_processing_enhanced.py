"""Enhanced comprehensive tests for video processing functionality.

This test suite focuses on improving test coverage for core video processing
functionality and edge cases not covered by existing tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deep_brief.core.exceptions import (
    ErrorCode,
    FileValidationError,
    FrameExtractionError,
)
from deep_brief.core.video_processor import FrameInfo, VideoInfo, VideoProcessor
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
def mock_video_info(tmp_path):
    """Create mock VideoInfo for testing."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("fake video content")

    return VideoInfo(
        file_path=video_file,
        duration=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        format="mp4",
        size_mb=50.0,
        codec="h264",
    )


class TestVideoProcessorInitialization:
    """Test VideoProcessor initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test processor initialization with default config."""
        processor = VideoProcessor()
        assert processor.config is not None
        assert processor.temp_dir.exists()
        assert len(processor.supported_formats) > 0

    def test_initialization_with_custom_config(self, mock_config):
        """Test processor initialization with custom config."""
        processor = VideoProcessor(config=mock_config)
        assert processor.config == mock_config
        assert processor.supported_formats == ["mp4", "mov", "avi", "webm"]
        assert processor.temp_dir == mock_config.processing.temp_dir

    def test_temp_directory_creation(self, mock_config):
        """Test that temp directory is created on initialization."""
        # Use a non-existent temp dir
        temp_path = Path(tempfile.mkdtemp()) / "nested" / "temp"
        mock_config.processing.temp_dir = temp_path

        processor = VideoProcessor(config=mock_config)
        assert processor.temp_dir.exists()
        assert processor.temp_dir == temp_path


class TestVideoFileValidation:
    """Test comprehensive video file validation functionality."""

    def test_validate_file_success(self, video_processor, tmp_path):
        """Test successful file validation with proper metadata."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")

        # Mock ffmpeg.probe to return valid video metadata
        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                    },
                ],
                "format": {
                    "duration": "120.0",
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }

            result = video_processor.validate_file(test_file)

            assert isinstance(result, VideoInfo)
            assert result.file_path == test_file
            assert result.duration == 120.0
            assert result.width == 1920
            assert result.height == 1080
            assert result.fps == 30.0
            assert result.codec == "h264"

    def test_validate_file_with_fractional_fps(self, video_processor, tmp_path):
        """Test file validation with fractional frame rate."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "24000/1001",  # 23.976 fps
                    }
                ],
                "format": {
                    "duration": "120.0",
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }

            result = video_processor.validate_file(test_file)
            assert abs(result.fps - 23.976) < 0.001  # Close to 23.976

    def test_validate_file_zero_denominator_fps(self, video_processor, tmp_path):
        """Test file validation with zero denominator in fps."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/0",  # Zero denominator
                    }
                ],
                "format": {
                    "duration": "120.0",
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }

            result = video_processor.validate_file(test_file)
            assert result.fps == 0

    def test_validate_file_invalid_dimensions(self, video_processor, tmp_path):
        """Test file validation with invalid video dimensions."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 0,  # Invalid width
                        "height": -100,  # Invalid height
                        "r_frame_rate": "30/1",
                    }
                ],
                "format": {
                    "duration": "120.0",
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }

            with pytest.raises(FileValidationError) as exc_info:
                video_processor.validate_file(test_file)

            assert exc_info.value.error_code == ErrorCode.FILE_CORRUPTED

    def test_validate_file_missing_metadata(self, video_processor, tmp_path):
        """Test file validation with missing required metadata."""
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"fake video content")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        # Missing width, height, fps
                    }
                ],
                "format": {
                    # Missing duration
                    "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                },
            }

            with pytest.raises(FileValidationError) as exc_info:
                video_processor.validate_file(test_file)

            assert exc_info.value.error_code == ErrorCode.FILE_CORRUPTED


class TestVideoProcessorUtilities:
    """Test utility methods of VideoProcessor."""

    def test_is_format_supported(self, video_processor):
        """Test format support checking."""
        assert video_processor.is_format_supported("test.mp4") is True
        assert (
            video_processor.is_format_supported("test.MP4") is True
        )  # Case insensitive
        assert video_processor.is_format_supported("test.mov") is True
        assert video_processor.is_format_supported("test.mkv") is False

    def test_get_supported_formats(self, video_processor):
        """Test getting supported formats list."""
        formats = video_processor.get_supported_formats()
        assert isinstance(formats, list)
        assert "mp4" in formats
        assert "mov" in formats
        assert "avi" in formats
        assert "webm" in formats

    def test_get_supported_formats_is_copy(self, video_processor):
        """Test that get_supported_formats returns a copy."""
        formats1 = video_processor.get_supported_formats()
        formats2 = video_processor.get_supported_formats()

        # Modify one list
        formats1.append("test")

        # Other list should be unchanged
        assert "test" not in formats2
        assert len(formats1) != len(formats2)

    def test_check_disk_space(self, video_processor):
        """Test disk space checking functionality."""
        # Test with reasonable amount (should pass)
        assert video_processor._check_disk_space(1.0) is True  # 1MB

        # Test with huge amount (should fail on most systems)
        # Using a very large number that would exceed available space
        assert video_processor._check_disk_space(1024 * 1024 * 1024) is False  # 1TB

    def test_check_ffmpeg_available(self, video_processor):
        """Test FFmpeg availability checking."""
        # This will depend on the test environment
        # Just test that it returns a boolean
        result = video_processor._check_ffmpeg_available()
        assert isinstance(result, bool)

    def test_get_quality_value(self, video_processor):
        """Test quality value conversion."""
        # Test default quality
        quality = video_processor._get_quality_value()
        assert 2 <= quality <= 31  # Valid ffmpeg quality range

        # Test with mock config
        with patch.object(video_processor.config.output, "frame_quality", 100):
            quality = video_processor._get_quality_value()
            assert quality == 2  # Highest quality

        with patch.object(video_processor.config.output, "frame_quality", 1):
            quality = video_processor._get_quality_value()
            assert quality == 31  # Lowest quality


class TestFrameExtraction:
    """Test frame extraction functionality."""

    def test_extract_frame_from_scene_success(
        self, video_processor, mock_video_info, tmp_path
    ):
        """Test successful frame extraction from a scene."""
        output_dir = tmp_path / "frames"

        # Mock ffmpeg operations
        with (
            patch("ffmpeg.input") as mock_input,
            patch("ffmpeg.output") as mock_output,
            patch("ffmpeg.overwrite_output") as mock_overwrite,
            patch("ffmpeg.run") as mock_run,
            patch("ffmpeg.filter") as mock_filter,
        ):
            # Set up mock chain
            mock_input.return_value = MagicMock()
            mock_filter.return_value = MagicMock()
            mock_output.return_value = MagicMock()
            mock_overwrite.return_value = MagicMock()

            # Create expected output file
            expected_path = output_dir / "scene_001_frame_15.00s.jpg"
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_bytes(b"fake jpeg content")

            result = video_processor.extract_frame_from_scene(
                video_info=mock_video_info,
                scene_start=10.0,
                scene_end=20.0,
                scene_number=1,
                output_dir=output_dir,
            )

            assert isinstance(result, FrameInfo)
            assert result.frame_path == expected_path
            assert result.timestamp == 15.0  # Middle of scene
            assert result.scene_number == 1
            assert result.width == mock_video_info.width
            assert result.height == mock_video_info.height

    def test_extract_frame_invalid_scene_times(self, video_processor, mock_video_info):
        """Test frame extraction with invalid scene times."""
        # End time before start time
        with pytest.raises(FrameExtractionError) as exc_info:
            video_processor.extract_frame_from_scene(
                video_info=mock_video_info,
                scene_start=20.0,
                scene_end=10.0,  # Invalid: end before start
                scene_number=1,
            )

        assert exc_info.value.error_code == ErrorCode.FRAME_EXTRACTION_FAILED

    def test_extract_frame_scene_beyond_duration(
        self, video_processor, mock_video_info, tmp_path
    ):
        """Test frame extraction when scene extends beyond video duration."""
        output_dir = tmp_path / "frames"

        with (
            patch("ffmpeg.input") as mock_input,
            patch("ffmpeg.output") as mock_output,
            patch("ffmpeg.overwrite_output") as mock_overwrite,
            patch("ffmpeg.run") as mock_run,
        ):
            mock_input.return_value = MagicMock()
            mock_output.return_value = MagicMock()
            mock_overwrite.return_value = MagicMock()

            # Create expected output file
            expected_path = output_dir / "scene_001_frame_105.00s.jpg"
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_bytes(b"fake jpeg content")

            # Scene extends beyond video duration (120s)
            result = video_processor.extract_frame_from_scene(
                video_info=mock_video_info,
                scene_start=100.0,
                scene_end=150.0,  # Beyond video duration
                scene_number=1,
                output_dir=output_dir,
            )

            # Should use middle of adjusted scene
            assert result.timestamp == 105.0

    def test_extract_frames_from_scenes_success(
        self, video_processor, mock_video_info, tmp_path
    ):
        """Test successful extraction of frames from multiple scenes."""
        scenes = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),
            (60.0, 90.0, 3),
        ]

        # Mock individual frame extraction
        with patch.object(video_processor, "extract_frame_from_scene") as mock_extract:
            mock_frames = []
            for i, (start, end, scene_num) in enumerate(scenes):
                frame_path = tmp_path / f"frame_{i}.jpg"
                frame_path.write_bytes(b"fake content")
                mock_frame = FrameInfo(
                    frame_path=frame_path,
                    timestamp=start + (end - start) / 2,
                    scene_number=scene_num,
                    width=1920,
                    height=1080,
                    size_kb=100.0,
                    format="jpg",
                )
                mock_frames.append(mock_frame)

            mock_extract.side_effect = mock_frames

            result = video_processor.extract_frames_from_scenes(
                video_info=mock_video_info, scenes=scenes
            )

            assert len(result) == 3
            assert all(isinstance(frame, FrameInfo) for frame in result)
            assert mock_extract.call_count == 3

    def test_extract_frames_from_scenes_empty_list(
        self, video_processor, mock_video_info
    ):
        """Test extraction with empty scenes list."""
        result = video_processor.extract_frames_from_scenes(
            video_info=mock_video_info, scenes=[]
        )

        assert result == []

    def test_extract_frames_partial_failure(
        self, video_processor, mock_video_info, tmp_path
    ):
        """Test partial failure during batch frame extraction."""
        scenes = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),  # This will fail
            (60.0, 90.0, 3),
        ]

        def mock_extract_side_effect(
            video_info, start_time, end_time, scene_number, **kwargs
        ):
            if scene_number == 2:
                raise FrameExtractionError(
                    "Frame extraction failed", scene_number=scene_number
                )

            frame_path = tmp_path / f"frame_{scene_number}.jpg"
            frame_path.write_bytes(b"fake content")
            return FrameInfo(
                frame_path=frame_path,
                timestamp=start_time + (end_time - start_time) / 2,
                scene_number=scene_number,
                width=1920,
                height=1080,
                size_kb=100.0,
                format="jpg",
            )

        with patch.object(video_processor, "extract_frame_from_scene") as mock_extract:
            mock_extract.side_effect = mock_extract_side_effect

            result = video_processor.extract_frames_from_scenes(
                video_info=mock_video_info, scenes=scenes
            )

            # Should get 2 successful frames (scenes 1 and 3)
            assert len(result) == 2
            assert result[0].scene_number == 1
            assert result[1].scene_number == 3


class TestVideoProcessorCleanup:
    """Test temporary file cleanup functionality."""

    def test_cleanup_temp_files_enabled(self, video_processor, tmp_path):
        """Test cleanup when enabled."""
        # Create some temp files
        temp_file1 = video_processor.temp_dir / "temp1.wav"
        temp_file2 = video_processor.temp_dir / "temp2.jpg"
        temp_subdir = video_processor.temp_dir / "subdir"

        temp_file1.write_text("temp content")
        temp_file2.write_text("temp content")
        temp_subdir.mkdir()

        # Ensure cleanup is enabled
        video_processor.config.processing.cleanup_temp_files = True

        video_processor.cleanup_temp_files()

        # Files should be removed
        assert not temp_file1.exists()
        assert not temp_file2.exists()
        assert not temp_subdir.exists()  # Empty directory removed

    def test_cleanup_temp_files_disabled(self, video_processor):
        """Test cleanup when disabled."""
        # Create some temp files
        temp_file = video_processor.temp_dir / "temp.wav"
        temp_file.write_text("temp content")

        # Disable cleanup
        video_processor.config.processing.cleanup_temp_files = False

        video_processor.cleanup_temp_files()

        # File should still exist
        assert temp_file.exists()

    def test_cleanup_nonexistent_temp_dir(self, video_processor):
        """Test cleanup when temp directory doesn't exist."""
        # Remove temp directory
        import shutil

        shutil.rmtree(video_processor.temp_dir)

        # Should not raise an error
        video_processor.cleanup_temp_files()

    def test_cleanup_handles_permission_errors(self, video_processor):
        """Test that cleanup handles permission errors gracefully."""
        temp_file = video_processor.temp_dir / "temp.wav"
        temp_file.write_text("temp content")

        # Mock permission error
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            # Should not raise an error, just log warning
            video_processor.cleanup_temp_files()


# Integration tests would go here
class TestVideoProcessorIntegration:
    """Test integration scenarios for VideoProcessor."""

    def test_validate_and_extract_workflow(self, video_processor, tmp_path):
        """Test complete workflow from validation to frame extraction."""
        # This would test the integration between validation and frame extraction
        # but requires more complex mocking of ffmpeg operations
        pass

    def test_error_recovery_scenarios(self, video_processor):
        """Test error recovery in various scenarios."""
        # Test that processor can recover from various error conditions
        pass
