"""Tests for scene detector functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from deep_brief.core.scene_detector import Scene, SceneDetectionResult, SceneDetector
from deep_brief.core.video_processor import VideoInfo
from deep_brief.utils.config import (
    DeepBriefConfig,
    ProcessingConfig,
    SceneDetectionConfig,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = DeepBriefConfig(
        processing=ProcessingConfig(
            temp_dir=Path(tempfile.mkdtemp()), cleanup_temp_files=True
        ),
        scene_detection=SceneDetectionConfig(
            method="threshold",
            threshold=0.4,
            min_scene_duration=2.0,
            fallback_interval=30.0,
        ),
    )
    return config


@pytest.fixture
def scene_detector(mock_config):
    """Create SceneDetector instance for testing."""
    return SceneDetector(config=mock_config)


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


class TestScene:
    """Test Scene model functionality."""

    def test_scene_creation(self):
        """Test Scene object creation and properties."""
        scene = Scene(
            start_time=10.5,
            end_time=45.2,
            duration=34.7,
            scene_number=1,
            confidence=0.8,
        )

        assert scene.start_time == 10.5
        assert scene.end_time == 45.2
        assert scene.duration == 34.7
        assert scene.scene_number == 1
        assert scene.confidence == 0.8

    def test_scene_time_formatting(self):
        """Test time formatting properties."""
        scene = Scene(
            start_time=3661.123,  # 1:01:01.123
            end_time=3725.456,  # 1:02:05.456
            duration=64.333,
            scene_number=1,
        )

        assert scene.start_time_str == "01:01:01.123"
        assert scene.end_time_str == "01:02:05.456"

    def test_scene_time_formatting_short(self):
        """Test time formatting for short durations."""
        scene = Scene(
            start_time=0.0,
            end_time=65.5,  # 1:05.500
            duration=65.5,
            scene_number=1,
        )

        assert scene.start_time_str == "00:00:00.000"
        assert scene.end_time_str == "00:01:05.500"


class TestSceneDetectionResult:
    """Test SceneDetectionResult model functionality."""

    def test_scene_detection_result_creation(self):
        """Test SceneDetectionResult creation."""
        scenes = [
            Scene(start_time=0.0, end_time=30.0, duration=30.0, scene_number=1),
            Scene(start_time=30.0, end_time=60.0, duration=30.0, scene_number=2),
            Scene(start_time=60.0, end_time=90.0, duration=30.0, scene_number=3),
        ]

        result = SceneDetectionResult(
            scenes=scenes,
            total_scenes=3,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=90.0,
            average_scene_duration=30.0,
        )

        assert result.total_scenes == 3
        assert result.detection_method == "threshold"
        assert result.threshold_used == 0.4
        assert result.video_duration == 90.0
        assert result.average_scene_duration == 30.0

    def test_scene_boundaries_property(self):
        """Test scene boundaries property."""
        scenes = [
            Scene(start_time=0.0, end_time=30.0, duration=30.0, scene_number=1),
            Scene(start_time=30.0, end_time=60.0, duration=30.0, scene_number=2),
            Scene(start_time=60.0, end_time=90.0, duration=30.0, scene_number=3),
        ]

        result = SceneDetectionResult(
            scenes=scenes,
            total_scenes=3,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=90.0,
            average_scene_duration=30.0,
        )

        # Should include start and all scene end times except the last
        expected_boundaries = [0.0, 30.0, 60.0]
        assert result.scene_boundaries == expected_boundaries


class TestSceneDetector:
    """Test SceneDetector class initialization and configuration."""

    def test_init_creates_temp_directory(self, mock_config):
        """Test that SceneDetector creates temp directory on init."""
        detector = SceneDetector(config=mock_config)
        assert detector.temp_dir.exists()
        assert detector.config.scene_detection.method == "threshold"
        assert detector.config.scene_detection.threshold == 0.4

    def test_init_with_adaptive_config(self, mock_config):
        """Test initialization with adaptive configuration."""
        mock_config.scene_detection.method = "adaptive"
        mock_config.scene_detection.threshold = 0.3

        detector = SceneDetector(config=mock_config)
        assert detector.config.scene_detection.method == "adaptive"
        assert detector.config.scene_detection.threshold == 0.3


class TestThresholdSceneDetection:
    """Test threshold-based scene detection."""

    @patch("deep_brief.core.scene_detector.SceneDetector._run_scene_detection")
    def test_detect_scenes_threshold_success(
        self, mock_run_detection, scene_detector, mock_video_info
    ):
        """Test successful threshold scene detection."""
        # Mock scene timestamps
        mock_run_detection.return_value = [0.0, 30.5, 65.2, 95.8]

        result = scene_detector.detect_scenes(mock_video_info)

        assert isinstance(result, SceneDetectionResult)
        assert result.detection_method == "threshold"
        assert result.threshold_used == 0.4
        assert result.total_scenes == 4
        assert len(result.scenes) == 4

        # Check first scene
        first_scene = result.scenes[0]
        assert first_scene.start_time == 0.0
        assert first_scene.end_time == 30.5
        assert first_scene.duration == 30.5
        assert first_scene.scene_number == 1

    @patch("deep_brief.core.scene_detector.SceneDetector._run_scene_detection")
    @patch("deep_brief.core.scene_detector.SceneDetector._fallback_scene_detection")
    def test_detect_scenes_threshold_fallback(
        self, mock_fallback, mock_run_detection, scene_detector, mock_video_info
    ):
        """Test fallback when threshold detection finds no scenes."""
        # Mock no scenes detected
        mock_run_detection.return_value = [0.0]  # Only start time

        mock_fallback_result = SceneDetectionResult(
            scenes=[
                Scene(start_time=0.0, end_time=30.0, duration=30.0, scene_number=1),
                Scene(start_time=30.0, end_time=60.0, duration=30.0, scene_number=2),
            ],
            total_scenes=2,
            detection_method="fallback",
            threshold_used=0.4,
            video_duration=120.0,
            average_scene_duration=30.0,
        )
        mock_fallback.return_value = mock_fallback_result

        result = scene_detector.detect_scenes(mock_video_info)

        assert result.detection_method == "fallback"
        mock_fallback.assert_called_once()

    @patch("deep_brief.core.scene_detector.SceneDetector._run_scene_detection")
    def test_detect_scenes_with_progress_callback(
        self, mock_run_detection, scene_detector, mock_video_info
    ):
        """Test scene detection with progress callback."""
        mock_run_detection.return_value = [0.0, 30.0, 60.0, 90.0]
        progress_callback = MagicMock()

        result = scene_detector.detect_scenes(mock_video_info, progress_callback)

        assert isinstance(result, SceneDetectionResult)
        # Progress callback should be passed to _run_scene_detection
        mock_run_detection.assert_called_once_with(
            mock_video_info, 0.4, progress_callback
        )


class TestAdaptiveSceneDetection:
    """Test adaptive scene detection."""

    def test_detect_scenes_adaptive_config(self, mock_config, mock_video_info):
        """Test adaptive scene detection configuration."""
        mock_config.scene_detection.method = "adaptive"
        detector = SceneDetector(config=mock_config)

        with patch.object(detector, "_detect_scenes_adaptive") as mock_adaptive:
            mock_adaptive.return_value = SceneDetectionResult(
                scenes=[],
                total_scenes=0,
                detection_method="adaptive",
                threshold_used=0.4,
                video_duration=120.0,
                average_scene_duration=0.0,
            )

            detector.detect_scenes(mock_video_info)
            mock_adaptive.assert_called_once_with(mock_video_info, None)

    @patch("deep_brief.core.scene_detector.SceneDetector._run_scene_detection")
    def test_adaptive_tries_multiple_thresholds(
        self, mock_run_detection, mock_config, mock_video_info
    ):
        """Test that adaptive method tries multiple thresholds."""
        mock_config.scene_detection.method = "adaptive"
        detector = SceneDetector(config=mock_config)

        # Mock different results for different thresholds
        mock_run_detection.side_effect = [
            [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],  # Too many scenes (0.2 threshold)
            [0.0, 30.0, 60.0, 90.0],  # Good result (0.4 threshold)
            [0.0, 60.0],  # Too few scenes (0.6 threshold)
            [0.0],  # No scenes (0.8 threshold)
        ]

        result = detector._detect_scenes_adaptive(mock_video_info)

        # Should have called _run_scene_detection multiple times
        assert mock_run_detection.call_count >= 2
        assert result.detection_method == "adaptive"
        assert result.total_scenes == 4  # Should pick the good result


class TestSceneDetectionExecution:
    """Test actual scene detection execution."""

    @patch("ffmpeg.run")
    def test_run_scene_detection_success(
        self, mock_run, scene_detector, mock_video_info
    ):
        """Test successful scene detection run."""
        # Mock ffmpeg output with scene detection info
        mock_result = MagicMock()
        mock_result.stderr = b"lavfi.scene_score=0.5 pts_time:30.5\nlavfi.scene_score=0.6 pts_time:65.2\n"
        mock_run.return_value = mock_result

        timestamps = scene_detector._run_scene_detection(mock_video_info, 0.4, None)

        # Should include start time plus detected times
        expected_times = [0.0, 30.5, 65.2]
        assert timestamps == expected_times

    @patch("ffmpeg.run")
    def test_run_scene_detection_ffmpeg_error(
        self, mock_run, scene_detector, mock_video_info
    ):
        """Test handling of ffmpeg errors."""
        mock_run.side_effect = ffmpeg.Error("ffmpeg", "stdout", "stderr_output")

        with pytest.raises(RuntimeError, match="Scene detection failed"):
            scene_detector._run_scene_detection(mock_video_info, 0.4, None)

    def test_parse_scene_timestamps(self, scene_detector):
        """Test parsing of scene timestamps from ffmpeg output."""
        ffmpeg_output = """
        [scdet @ 0x123] lavfi.scene_score=0.456789 pts_time:30.500
        [scdet @ 0x124] lavfi.scene_score=0.123456 pts_time:65.250
        [Parsed_scdet_0 @ 0x125] scene_score=0.789 time=95.750
        """

        timestamps = scene_detector._parse_scene_timestamps(ffmpeg_output)

        # Should include start time plus parsed times
        expected_times = [0.0, 30.5, 65.25, 95.75]
        assert timestamps == expected_times

    def test_parse_scene_timestamps_no_matches(self, scene_detector):
        """Test parsing when no scene timestamps found."""
        ffmpeg_output = "No scene detection output here"

        timestamps = scene_detector._parse_scene_timestamps(ffmpeg_output)

        # Should only have start time
        assert timestamps == [0.0]


class TestSceneCreation:
    """Test scene creation from timestamps."""

    def test_create_scenes_from_timestamps(self, scene_detector):
        """Test creating scenes from timestamps."""
        timestamps = [0.0, 30.5, 65.2, 95.8]
        total_duration = 120.0

        scenes = scene_detector._create_scenes_from_timestamps(
            timestamps, total_duration
        )

        assert len(scenes) == 4

        # Check first scene
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == 30.5
        assert scenes[0].duration == 30.5
        assert scenes[0].scene_number == 1

        # Check last scene (should end at video duration)
        assert scenes[-1].start_time == 95.8
        assert scenes[-1].end_time == 120.0
        assert scenes[-1].duration == pytest.approx(24.2)
        assert scenes[-1].scene_number == 4

    def test_create_scenes_filters_short_scenes(self, scene_detector):
        """Test that very short scenes are filtered out."""
        timestamps = [0.0, 0.2, 30.5, 30.7, 65.0]  # 0.2s and 0.2s scenes
        total_duration = 90.0

        scenes = scene_detector._create_scenes_from_timestamps(
            timestamps, total_duration
        )

        # Short scenes should be filtered out
        assert len(scenes) == 2  # Only 0.0-30.5 and 65.0-90.0 remain
        assert scenes[0].duration == 30.5
        assert scenes[1].duration == 25.0

    def test_filter_scenes_by_duration(self, scene_detector):
        """Test filtering scenes by minimum duration."""
        scenes = [
            Scene(
                start_time=0.0, end_time=1.0, duration=1.0, scene_number=1
            ),  # Too short
            Scene(
                start_time=1.0, end_time=5.0, duration=4.0, scene_number=2
            ),  # Long enough
            Scene(
                start_time=5.0, end_time=6.5, duration=1.5, scene_number=3
            ),  # Too short
            Scene(
                start_time=6.5, end_time=10.0, duration=3.5, scene_number=4
            ),  # Long enough
        ]

        filtered = scene_detector._filter_scenes_by_duration(scenes)

        assert len(filtered) == 2
        assert filtered[0].duration == 4.0
        assert filtered[1].duration == 3.5
        # Scene numbers should be renumbered
        assert filtered[0].scene_number == 1
        assert filtered[1].scene_number == 2


class TestFallbackDetection:
    """Test fallback scene detection."""

    def test_fallback_scene_detection(self, scene_detector, mock_video_info):
        """Test fallback scene detection with fixed intervals."""
        result = scene_detector._fallback_scene_detection(mock_video_info, 0.4)

        assert result.detection_method == "fallback"
        assert result.threshold_used == 0.4
        assert result.video_duration == 120.0
        assert result.total_scenes == 4  # 120s / 30s = 4 scenes

        # Check scene intervals
        for i, scene in enumerate(result.scenes):
            expected_start = i * 30.0
            expected_end = min((i + 1) * 30.0, 120.0)
            assert scene.start_time == expected_start
            assert scene.end_time == expected_end
            assert scene.confidence == 0.3  # Fallback confidence

    def test_fallback_scene_detection_custom_interval(
        self, mock_config, mock_video_info
    ):
        """Test fallback with custom interval."""
        mock_config.scene_detection.fallback_interval = 60.0  # 1 minute intervals
        detector = SceneDetector(config=mock_config)

        result = detector._fallback_scene_detection(mock_video_info, 0.4)

        assert result.total_scenes == 2  # 120s / 60s = 2 scenes
        assert result.scenes[0].duration == 60.0
        assert result.scenes[1].duration == 60.0


class TestSceneSummary:
    """Test scene summary and statistics."""

    def test_get_scene_summary_with_scenes(self, scene_detector):
        """Test scene summary generation."""
        scenes = [
            Scene(start_time=0.0, end_time=25.0, duration=25.0, scene_number=1),
            Scene(start_time=25.0, end_time=60.0, duration=35.0, scene_number=2),
            Scene(start_time=60.0, end_time=80.0, duration=20.0, scene_number=3),
        ]

        result = SceneDetectionResult(
            scenes=scenes,
            total_scenes=3,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=80.0,
            average_scene_duration=26.67,
        )

        summary = scene_detector.get_scene_summary(result)

        assert summary["total_scenes"] == 3
        assert summary["average_duration"] == 26.67
        assert summary["shortest_scene"] == 20.0
        assert summary["longest_scene"] == 35.0
        assert summary["detection_method"] == "threshold"
        assert summary["threshold_used"] == 0.4
        assert summary["scene_boundaries"] == [0.0, 25.0, 60.0]

    def test_get_scene_summary_no_scenes(self, scene_detector):
        """Test scene summary with no scenes."""
        result = SceneDetectionResult(
            scenes=[],
            total_scenes=0,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=120.0,
            average_scene_duration=0.0,
        )

        summary = scene_detector.get_scene_summary(result)

        assert summary["total_scenes"] == 0
        assert summary["average_duration"] == 0.0
        assert summary["shortest_scene"] == 0.0
        assert summary["longest_scene"] == 0.0


class TestProgressTracking:
    """Test progress tracking during scene detection."""

    def test_run_with_progress_success(self, scene_detector):
        """Test progress tracking with successful execution."""
        mock_stream = MagicMock()
        progress_callback = MagicMock()

        # Mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, None, 0]  # Running, then complete

        # Create a function that returns empty bytes indefinitely after the real data
        readline_values = [
            b"time=00:00:30.00 bitrate=N/A speed= 1.0x\n",
            b"time=00:01:00.00 bitrate=N/A speed= 1.0x\n",
        ]
        readline_index = 0

        def mock_readline():
            nonlocal readline_index
            if readline_index < len(readline_values):
                result = readline_values[readline_index]
                readline_index += 1
                return result
            return b""  # Return empty bytes forever after

        mock_process.stderr.readline.side_effect = mock_readline
        mock_process.wait.return_value = None
        mock_process.stderr.read.return_value = b"scene detection output"

        with patch("ffmpeg.run_async", return_value=mock_process):
            result = scene_detector._run_with_progress(
                mock_stream, 120.0, progress_callback
            )

        # Should call progress callback for each time update plus final
        assert progress_callback.call_count >= 1
        # Final call should be 1.0 (100% complete)
        progress_callback.assert_called_with(1.0)
        assert hasattr(result, "stderr")
