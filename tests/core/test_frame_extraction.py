"""Tests for frame extraction functionality."""

from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from deep_brief.core.video_processor import FrameInfo, VideoInfo, VideoProcessor


@pytest.fixture
def video_processor():
    """Create VideoProcessor instance for testing."""
    return VideoProcessor()


@pytest.fixture
def mock_video_info(tmp_path):
    """Create mock VideoInfo for testing."""
    test_video = tmp_path / "test_video.mp4"
    test_video.write_text("mock video content")

    return VideoInfo(
        file_path=test_video,
        duration=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        format="mp4",
        size_mb=50.0,
        codec="h264",
    )


class TestFrameExtraction:
    """Test frame extraction from scenes."""

    @patch("ffmpeg.run")
    def test_extract_frame_from_scene_success(
        self, mock_run, video_processor, mock_video_info, tmp_path
    ):
        """Test successful frame extraction from a scene."""
        output_dir = tmp_path / "frames"

        # Mock ffmpeg.run to succeed
        mock_run.return_value = None

        # Create expected output file
        expected_output = output_dir / "scene_001_frame_15.00s.jpg"
        output_dir.mkdir(parents=True)
        expected_output.write_bytes(b"mock jpeg content")

        frame_info = video_processor.extract_frame_from_scene(
            mock_video_info,
            scene_start=10.0,
            scene_end=20.0,
            scene_number=1,
            output_dir=output_dir,
        )

        assert isinstance(frame_info, FrameInfo)
        assert frame_info.frame_path == expected_output
        assert frame_info.timestamp == 15.0  # Middle of scene
        assert frame_info.scene_number == 1
        assert frame_info.width == 1920
        assert frame_info.height == 1080
        assert frame_info.format == "jpg"
        assert frame_info.size_kb > 0

        # Verify ffmpeg was called correctly
        mock_run.assert_called_once()

    @patch("ffmpeg.run")
    def test_extract_frame_with_progress_callback(
        self, mock_run, video_processor, mock_video_info, tmp_path
    ):
        """Test frame extraction with progress callback."""
        output_dir = tmp_path / "frames"
        progress_callback = MagicMock()

        # Mock ffmpeg.run to succeed
        mock_run.return_value = None

        # Create expected output file
        expected_output = output_dir / "scene_002_frame_45.00s.jpg"
        output_dir.mkdir(parents=True)
        expected_output.write_bytes(b"mock jpeg content")

        video_processor.extract_frame_from_scene(
            mock_video_info,
            scene_start=30.0,
            scene_end=60.0,
            scene_number=2,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called
        progress_callback.assert_called_once_with(1.0)

    @patch("ffmpeg.run")
    def test_extract_frame_default_output_dir(
        self, mock_run, video_processor, mock_video_info
    ):
        """Test frame extraction with default output directory."""
        mock_run.return_value = None

        # Create expected output file in temp dir
        expected_dir = video_processor.temp_dir / "frames"
        expected_output = expected_dir / "scene_003_frame_75.50s.jpg"
        expected_dir.mkdir(parents=True, exist_ok=True)
        expected_output.write_bytes(b"mock jpeg content")

        frame_info = video_processor.extract_frame_from_scene(
            mock_video_info,
            scene_start=70.0,
            scene_end=81.0,
            scene_number=3,
        )

        assert frame_info.frame_path == expected_output
        assert frame_info.timestamp == 75.5

    @patch("ffmpeg.run")
    def test_extract_frame_ffmpeg_error(
        self, mock_run, video_processor, mock_video_info, tmp_path
    ):
        """Test handling of ffmpeg errors during frame extraction."""
        mock_run.side_effect = ffmpeg.Error("ffmpeg", "stdout", "stderr_output")

        with pytest.raises(RuntimeError, match="Frame extraction failed"):
            video_processor.extract_frame_from_scene(
                mock_video_info,
                scene_start=10.0,
                scene_end=20.0,
                scene_number=1,
                output_dir=tmp_path,
            )

    @patch("ffmpeg.run")
    def test_extract_frame_output_file_not_created(
        self, mock_run, video_processor, mock_video_info, tmp_path
    ):
        """Test error when output file is not created."""
        mock_run.return_value = None  # ffmpeg succeeds but no file created

        with pytest.raises(RuntimeError, match="output file not found"):
            video_processor.extract_frame_from_scene(
                mock_video_info,
                scene_start=10.0,
                scene_end=20.0,
                scene_number=1,
                output_dir=tmp_path,
            )

    def test_extract_frames_from_scenes_empty_list(
        self, video_processor, mock_video_info
    ):
        """Test handling of empty scenes list."""
        result = video_processor.extract_frames_from_scenes(mock_video_info, [])

        assert result == []

    @patch("deep_brief.core.video_processor.VideoProcessor.extract_frame_from_scene")
    def test_extract_frames_from_scenes_success(
        self, mock_extract_frame, video_processor, mock_video_info, tmp_path
    ):
        """Test successful extraction of frames from multiple scenes."""
        # Mock individual frame extraction
        mock_frame_info = FrameInfo(
            frame_path=tmp_path / "test_frame.jpg",
            timestamp=15.0,
            scene_number=1,
            width=1920,
            height=1080,
            size_kb=100.0,
        )
        mock_extract_frame.return_value = mock_frame_info

        scenes = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),
            (60.0, 90.0, 3),
        ]

        progress_callback = MagicMock()

        result = video_processor.extract_frames_from_scenes(
            mock_video_info, scenes, tmp_path, progress_callback
        )

        assert len(result) == 3
        assert all(isinstance(frame, FrameInfo) for frame in result)

        # Verify extract_frame_from_scene was called for each scene
        assert mock_extract_frame.call_count == 3

        # Verify progress callback was called correctly
        expected_progress_calls = [((1 / 3,),), ((2 / 3,),), ((3 / 3,),)]
        assert progress_callback.call_args_list == expected_progress_calls

    @patch("deep_brief.core.video_processor.VideoProcessor.extract_frame_from_scene")
    def test_extract_frames_partial_failure(
        self, mock_extract_frame, video_processor, mock_video_info, tmp_path
    ):
        """Test handling of partial failures during batch frame extraction."""

        # Mock some successes and some failures
        def side_effect(*args, **kwargs):  # noqa: ARG001
            scene_number = args[3]  # scene_number is 4th argument
            if scene_number == 2:
                raise RuntimeError("Frame extraction failed")

            return FrameInfo(
                frame_path=tmp_path / f"scene_{scene_number:03d}.jpg",
                timestamp=15.0,
                scene_number=scene_number,
                width=1920,
                height=1080,
                size_kb=100.0,
            )

        mock_extract_frame.side_effect = side_effect

        scenes = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),  # This will fail
            (60.0, 90.0, 3),
        ]

        result = video_processor.extract_frames_from_scenes(
            mock_video_info, scenes, tmp_path
        )

        # Should get 2 successful extractions (scenes 1 and 3)
        assert len(result) == 2
        assert result[0].scene_number == 1
        assert result[1].scene_number == 3


class TestQualityConversion:
    """Test quality value conversion."""

    def test_get_quality_value_default(self, video_processor):
        """Test quality value conversion with default settings."""
        quality_value = video_processor._get_quality_value()

        # Default is 80%, should map to a reasonable ffmpeg quality value
        assert isinstance(quality_value, int)
        assert 2 <= quality_value <= 31

    def test_get_quality_value_bounds(self, video_processor):
        """Test quality value bounds."""
        # Test that quality values are always within valid range
        for quality_percent in [1, 50, 80, 100]:
            # Mock the config value
            video_processor.config.output.frame_quality = quality_percent
            quality_value = video_processor._get_quality_value()
            assert 2 <= quality_value <= 31
            assert isinstance(quality_value, int)


class TestFrameInfo:
    """Test FrameInfo model."""

    def test_frame_info_creation(self, tmp_path):
        """Test FrameInfo model creation."""
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"test content")

        frame_info = FrameInfo(
            frame_path=frame_path,
            timestamp=15.5,
            scene_number=1,
            width=1920,
            height=1080,
            size_kb=125.6,
            format="jpg",
        )

        assert frame_info.frame_path == frame_path
        assert frame_info.timestamp == 15.5
        assert frame_info.scene_number == 1
        assert frame_info.width == 1920
        assert frame_info.height == 1080
        assert frame_info.size_kb == 125.6
        assert frame_info.format == "jpg"

    def test_frame_info_default_format(self, tmp_path):
        """Test FrameInfo with default format."""
        frame_path = tmp_path / "test_frame.jpg"

        frame_info = FrameInfo(
            frame_path=frame_path,
            timestamp=30.0,
            scene_number=2,
            width=1280,
            height=720,
            size_kb=95.3,
        )

        assert frame_info.format == "jpg"  # Default value
