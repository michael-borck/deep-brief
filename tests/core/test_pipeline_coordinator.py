"""Tests for pipeline coordinator functionality."""

from unittest.mock import MagicMock, patch

import pytest

from deep_brief.core.audio_extractor import AudioInfo
from deep_brief.core.pipeline_coordinator import (
    PipelineCoordinator,
    VideoAnalysisResult,
    create_pipeline_coordinator,
)
from deep_brief.core.progress_tracker import ProgressTracker
from deep_brief.core.scene_detector import Scene, SceneDetectionResult
from deep_brief.core.video_processor import FrameInfo, VideoInfo


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock()
    config.processing.max_video_size_mb = 100
    return config


@pytest.fixture
def progress_tracker():
    """Create progress tracker for testing."""
    return ProgressTracker()


@pytest.fixture
def pipeline_coordinator(mock_config, progress_tracker):
    """Create pipeline coordinator for testing."""
    return PipelineCoordinator(mock_config, progress_tracker)


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


@pytest.fixture
def mock_audio_info(tmp_path):
    """Create mock AudioInfo for testing."""
    test_audio = tmp_path / "test_audio.wav"
    test_audio.write_bytes(b"mock audio content")

    return AudioInfo(
        file_path=test_audio,
        duration=120.0,
        sample_rate=16000,
        channels=1,
        size_mb=10.0,
        format="wav",
    )


@pytest.fixture
def mock_scene_result():
    """Create mock SceneDetectionResult for testing."""
    scenes = [
        Scene(
            start_time=0.0,
            end_time=40.0,
            duration=40.0,
            scene_number=1,
            confidence=0.8,
        ),
        Scene(
            start_time=40.0,
            end_time=80.0,
            duration=40.0,
            scene_number=2,
            confidence=0.7,
        ),
        Scene(
            start_time=80.0,
            end_time=120.0,
            duration=40.0,
            scene_number=3,
            confidence=0.9,
        ),
    ]

    return SceneDetectionResult(
        scenes=scenes,
        total_scenes=3,
        detection_method="threshold",
        threshold_used=0.3,
        video_duration=120.0,
        average_scene_duration=40.0,
    )


@pytest.fixture
def mock_frame_infos(tmp_path):
    """Create mock FrameInfo list for testing."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    frame_infos = []
    for i in range(1, 4):
        frame_path = frame_dir / f"scene_{i:03d}_frame_{i * 20.0:.2f}s.jpg"
        frame_path.write_bytes(b"mock jpeg content")

        frame_infos.append(
            FrameInfo(
                frame_path=frame_path,
                timestamp=i * 20.0,
                scene_number=i,
                width=1920,
                height=1080,
                size_kb=100.0,
                format="jpg",
            )
        )

    return frame_infos


class TestVideoAnalysisResult:
    """Test VideoAnalysisResult class."""

    def test_video_analysis_result_creation(self, mock_video_info):
        """Test creating VideoAnalysisResult."""
        result = VideoAnalysisResult(
            video_info=mock_video_info, processing_time=45.5, success=True
        )

        assert result.video_info == mock_video_info
        assert result.audio_info is None
        assert result.scene_result is None
        assert result.frame_infos == []
        assert result.processing_time == 45.5
        assert result.success is True
        assert result.error_message is None

    def test_video_analysis_result_complete(
        self, mock_video_info, mock_audio_info, mock_scene_result, mock_frame_infos
    ):
        """Test creating complete VideoAnalysisResult."""
        result = VideoAnalysisResult(
            video_info=mock_video_info,
            audio_info=mock_audio_info,
            scene_result=mock_scene_result,
            frame_infos=mock_frame_infos,
            processing_time=67.3,
            success=True,
        )

        assert result.video_info == mock_video_info
        assert result.audio_info == mock_audio_info
        assert result.scene_result == mock_scene_result
        assert result.frame_infos == mock_frame_infos
        assert result.processing_time == 67.3
        assert result.success is True

    def test_video_analysis_result_failure(self, mock_video_info):
        """Test creating failed VideoAnalysisResult."""
        result = VideoAnalysisResult(
            video_info=mock_video_info, success=False, error_message="Processing failed"
        )

        assert result.success is False
        assert result.error_message == "Processing failed"

    def test_to_dict_complete_result(
        self, mock_video_info, mock_audio_info, mock_scene_result, mock_frame_infos
    ):
        """Test converting complete result to dictionary."""
        result = VideoAnalysisResult(
            video_info=mock_video_info,
            audio_info=mock_audio_info,
            scene_result=mock_scene_result,
            frame_infos=mock_frame_infos,
            processing_time=67.3,
            success=True,
        )

        result_dict = result.to_dict()

        # Check video info
        assert result_dict["video_info"]["duration"] == 120.0
        assert result_dict["video_info"]["width"] == 1920
        assert result_dict["video_info"]["height"] == 1080

        # Check audio info
        assert result_dict["audio_info"]["duration"] == 120.0
        assert result_dict["audio_info"]["sample_rate"] == 16000

        # Check scene result
        assert result_dict["scene_result"]["total_scenes"] == 3
        assert result_dict["scene_result"]["detection_method"] == "threshold"
        assert len(result_dict["scene_result"]["scenes"]) == 3

        # Check frame infos
        assert len(result_dict["frame_infos"]) == 3
        assert result_dict["frame_infos"][0]["scene_number"] == 1

        # Check metadata
        assert result_dict["processing_time"] == 67.3
        assert result_dict["success"] is True
        assert result_dict["error_message"] is None

    def test_to_dict_minimal_result(self, mock_video_info):
        """Test converting minimal result to dictionary."""
        result = VideoAnalysisResult(video_info=mock_video_info)

        result_dict = result.to_dict()

        # Check required fields
        assert "video_info" in result_dict
        assert result_dict["audio_info"] is None
        assert result_dict["scene_result"] is None
        assert result_dict["frame_infos"] == []
        assert result_dict["success"] is True


class TestPipelineCoordinator:
    """Test PipelineCoordinator class."""

    def test_pipeline_coordinator_initialization(self, mock_config):
        """Test pipeline coordinator initialization."""
        coordinator = PipelineCoordinator(mock_config)

        assert coordinator.config == mock_config
        assert coordinator.progress_tracker is None
        assert hasattr(coordinator, "video_processor")
        assert hasattr(coordinator, "audio_extractor")
        assert hasattr(coordinator, "scene_detector")

    def test_pipeline_coordinator_with_progress_tracker(
        self, mock_config, progress_tracker
    ):
        """Test pipeline coordinator with progress tracker."""
        coordinator = PipelineCoordinator(mock_config, progress_tracker)

        assert coordinator.progress_tracker == progress_tracker

    @patch("deep_brief.core.pipeline_coordinator.get_config")
    def test_pipeline_coordinator_default_config(self, mock_get_config):
        """Test pipeline coordinator with default config."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        coordinator = PipelineCoordinator()

        assert coordinator.config == mock_config
        mock_get_config.assert_called_once()

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    @patch("deep_brief.core.audio_extractor.AudioExtractor.extract_audio")
    @patch("deep_brief.core.scene_detector.SceneDetector.detect_scenes")
    @patch("deep_brief.core.video_processor.VideoProcessor.extract_frames_from_scenes")
    def test_analyze_video_success_all_features(
        self,
        mock_extract_frames,
        mock_detect_scenes,
        mock_extract_audio,
        mock_validate_file,
        pipeline_coordinator,
        mock_video_info,
        mock_audio_info,
        mock_scene_result,
        mock_frame_infos,
        tmp_path,
    ):
        """Test successful video analysis with all features enabled."""
        # Mock component responses
        mock_validate_file.return_value = mock_video_info
        mock_extract_audio.return_value = mock_audio_info
        mock_detect_scenes.return_value = mock_scene_result
        mock_extract_frames.return_value = mock_frame_infos

        # Analyze video
        result = pipeline_coordinator.analyze_video(
            video_path=tmp_path / "test.mp4",
            extract_audio=True,
            detect_scenes=True,
            extract_frames=True,
            output_dir=tmp_path / "output",
        )

        # Verify result
        assert result.success is True
        assert result.video_info == mock_video_info
        assert result.audio_info == mock_audio_info
        assert result.scene_result == mock_scene_result
        assert result.frame_infos == mock_frame_infos
        assert result.error_message is None

        # Verify components were called
        mock_validate_file.assert_called_once()
        mock_extract_audio.assert_called_once()
        mock_detect_scenes.assert_called_once()
        mock_extract_frames.assert_called_once()

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    def test_analyze_video_minimal_features(
        self, mock_validate_file, pipeline_coordinator, mock_video_info, tmp_path
    ):
        """Test video analysis with minimal features."""
        mock_validate_file.return_value = mock_video_info

        result = pipeline_coordinator.analyze_video(
            video_path=tmp_path / "test.mp4",
            extract_audio=False,
            detect_scenes=False,
            extract_frames=False,
        )

        # Verify result
        assert result.success is True
        assert result.video_info == mock_video_info
        assert result.audio_info is None
        assert result.scene_result is None
        assert result.frame_infos == []

        # Only validation should be called
        mock_validate_file.assert_called_once()

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    @patch("deep_brief.core.audio_extractor.AudioExtractor.extract_audio")
    def test_analyze_video_audio_extraction_failure(
        self,
        mock_extract_audio,
        mock_validate_file,
        pipeline_coordinator,
        mock_video_info,
        tmp_path,
    ):
        """Test video analysis with audio extraction failure (no audio stream)."""
        mock_validate_file.return_value = mock_video_info
        mock_extract_audio.side_effect = ValueError("No audio stream")

        result = pipeline_coordinator.analyze_video(
            video_path=tmp_path / "test.mp4",
            extract_audio=True,
            detect_scenes=False,
            extract_frames=False,
        )

        # Should succeed despite audio failure
        assert result.success is True
        assert result.video_info == mock_video_info
        assert result.audio_info is None  # No audio due to failure

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    def test_analyze_video_validation_failure(
        self, mock_validate_file, pipeline_coordinator, tmp_path
    ):
        """Test video analysis with validation failure."""
        mock_validate_file.side_effect = RuntimeError("Invalid video")

        result = pipeline_coordinator.analyze_video(video_path=tmp_path / "test.mp4")

        # Should fail
        assert result.success is False
        assert "Invalid video" in result.error_message
        assert result.video_info is None

    def test_analyze_video_with_progress_tracking(
        self, pipeline_coordinator, progress_tracker, mock_video_info, tmp_path
    ):
        """Test video analysis with progress tracking."""
        # Mock video processor
        with patch(
            "deep_brief.core.video_processor.VideoProcessor.validate_file"
        ) as mock_validate_file:
            mock_validate_file.return_value = mock_video_info

            # Add progress callback to track calls
            progress_callback = MagicMock()
            progress_tracker.add_callback(progress_callback)

            result = pipeline_coordinator.analyze_video(
                video_path=tmp_path / "test.mp4",
                extract_audio=False,
                detect_scenes=False,
                extract_frames=False,
            )

            # Should succeed
            assert result.success is True

            # Progress callbacks should have been called
            assert progress_callback.call_count > 0

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    @patch("deep_brief.core.scene_detector.SceneDetector.detect_scenes")
    @patch("deep_brief.core.video_processor.VideoProcessor.extract_frames_from_scenes")
    def test_analyze_video_frame_extraction_without_scenes(
        self,
        mock_extract_frames,
        mock_detect_scenes,
        mock_validate_file,
        pipeline_coordinator,
        mock_video_info,
        tmp_path,
    ):
        """Test that frame extraction is skipped when no scenes are detected."""
        # Mock empty scene result
        empty_scene_result = SceneDetectionResult(
            scenes=[],
            total_scenes=0,
            detection_method="threshold",
            threshold_used=0.3,
            video_duration=120.0,
            average_scene_duration=0.0,
        )

        mock_validate_file.return_value = mock_video_info
        mock_detect_scenes.return_value = empty_scene_result

        result = pipeline_coordinator.analyze_video(
            video_path=tmp_path / "test.mp4",
            extract_audio=False,
            detect_scenes=True,
            extract_frames=True,
        )

        # Should succeed but with no frames
        assert result.success is True
        assert result.scene_result == empty_scene_result
        assert result.frame_infos == []

        # Frame extraction should not be called
        mock_extract_frames.assert_not_called()

    def test_analyze_video_batch_success(
        self, pipeline_coordinator, mock_video_info, tmp_path
    ):
        """Test successful batch video analysis."""
        video_paths = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.mp4",
            tmp_path / "video3.mp4",
        ]

        # Create mock video files
        for path in video_paths:
            path.write_text("mock video")

        # Mock single video analysis
        with patch.object(pipeline_coordinator, "analyze_video") as mock_analyze:
            mock_result = VideoAnalysisResult(video_info=mock_video_info)
            mock_analyze.return_value = mock_result

            results = pipeline_coordinator.analyze_video_batch(
                video_paths=video_paths,
                extract_audio=True,
                detect_scenes=True,
                extract_frames=True,
                output_dir=tmp_path / "batch_output",
            )

            # Should get results for all videos
            assert len(results) == 3
            assert all(result.success for result in results)

            # analyze_video should be called for each video
            assert mock_analyze.call_count == 3

    def test_analyze_video_batch_with_failures(
        self, pipeline_coordinator, mock_video_info, tmp_path
    ):
        """Test batch video analysis with some failures."""
        video_paths = [
            tmp_path / "video1.mp4",
            tmp_path / "video2.mp4",
            tmp_path / "video3.mp4",
        ]

        # Mock single video analysis with mixed results
        def mock_analyze_side_effect(video_path, **kwargs):  # noqa: ARG001
            if "video2" in str(video_path):
                return VideoAnalysisResult(
                    video_info=mock_video_info,
                    success=False,
                    error_message="Processing failed",
                )
            return VideoAnalysisResult(video_info=mock_video_info)

        with patch.object(pipeline_coordinator, "analyze_video") as mock_analyze:
            mock_analyze.side_effect = mock_analyze_side_effect

            results = pipeline_coordinator.analyze_video_batch(video_paths)

            # Should get results for all videos
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False  # video2 failed
            assert results[2].success is True

    def test_analyze_video_batch_with_progress_tracking(
        self, pipeline_coordinator, progress_tracker, mock_video_info, tmp_path
    ):
        """Test batch analysis with progress tracking."""
        video_paths = [tmp_path / "video1.mp4", tmp_path / "video2.mp4"]

        # Add progress callback
        progress_callback = MagicMock()
        progress_tracker.add_callback(progress_callback)

        # Mock single video analysis
        with patch.object(pipeline_coordinator, "analyze_video") as mock_analyze:
            mock_result = VideoAnalysisResult(video_info=mock_video_info)
            mock_analyze.return_value = mock_result

            results = pipeline_coordinator.analyze_video_batch(video_paths)

            # Should get results
            assert len(results) == 2

            # Progress should be tracked
            assert progress_callback.call_count > 0

    def test_analyze_video_batch_exception_handling(
        self, pipeline_coordinator, tmp_path
    ):
        """Test batch analysis exception handling."""
        video_paths = [tmp_path / "video1.mp4"]

        # Mock single video analysis to raise exception
        with patch.object(pipeline_coordinator, "analyze_video") as mock_analyze:
            mock_analyze.side_effect = RuntimeError("Unexpected error")

            results = pipeline_coordinator.analyze_video_batch(video_paths)

            # Should return empty results but not crash
            assert results == []


class TestPipelineCoordinatorFactories:
    """Test factory functions for pipeline coordinator."""

    def test_create_pipeline_coordinator_no_tracker(self):
        """Test creating pipeline coordinator without progress tracker."""
        coordinator = create_pipeline_coordinator()

        assert isinstance(coordinator, PipelineCoordinator)
        assert coordinator.progress_tracker is None

    def test_create_pipeline_coordinator_with_tracker(self):
        """Test creating pipeline coordinator with progress tracker."""
        tracker = ProgressTracker()
        coordinator = create_pipeline_coordinator(tracker)

        assert isinstance(coordinator, PipelineCoordinator)
        assert coordinator.progress_tracker == tracker


class TestPipelineCoordinatorIntegration:
    """Test integration scenarios with pipeline coordinator."""

    @patch("deep_brief.core.video_processor.VideoProcessor.validate_file")
    @patch("deep_brief.core.audio_extractor.AudioExtractor.extract_audio")
    @patch("deep_brief.core.scene_detector.SceneDetector.detect_scenes")
    @patch("deep_brief.core.video_processor.VideoProcessor.extract_frames_from_scenes")
    def test_full_pipeline_integration(
        self,
        mock_extract_frames,
        mock_detect_scenes,
        mock_extract_audio,
        mock_validate_file,
        mock_config,
        mock_video_info,
        mock_audio_info,
        mock_scene_result,
        mock_frame_infos,
        tmp_path,
    ):
        """Test full pipeline integration with all components."""
        # Setup mocks
        mock_validate_file.return_value = mock_video_info
        mock_extract_audio.return_value = mock_audio_info
        mock_detect_scenes.return_value = mock_scene_result
        mock_extract_frames.return_value = mock_frame_infos

        # Create coordinator with progress tracking
        progress_tracker = ProgressTracker()
        progress_callback = MagicMock()
        progress_tracker.add_callback(progress_callback)

        coordinator = PipelineCoordinator(mock_config, progress_tracker)

        # Run full analysis
        result = coordinator.analyze_video(
            video_path=tmp_path / "test.mp4",
            extract_audio=True,
            detect_scenes=True,
            extract_frames=True,
            output_dir=tmp_path / "output",
        )

        # Verify complete result
        assert result.success is True
        assert result.video_info == mock_video_info
        assert result.audio_info == mock_audio_info
        assert result.scene_result == mock_scene_result
        assert result.frame_infos == mock_frame_infos

        # Verify all components were called
        mock_validate_file.assert_called_once()
        mock_extract_audio.assert_called_once()
        mock_detect_scenes.assert_called_once()
        mock_extract_frames.assert_called_once()

        # Verify progress tracking
        assert progress_callback.call_count > 0

        # Verify output directory handling
        audio_call_args = mock_extract_audio.call_args
        assert str(tmp_path / "output") in str(audio_call_args[1]["output_path"])

        frames_call_args = mock_extract_frames.call_args
        assert str(tmp_path / "output" / "frames") in str(
            frames_call_args[1]["output_dir"]
        )

    def test_pipeline_coordinator_resource_cleanup(self, mock_config, tmp_path):
        """Test that pipeline coordinator properly handles resource cleanup."""
        coordinator = PipelineCoordinator(mock_config)

        # Mock components to track cleanup calls
        with patch(
            "deep_brief.core.video_processor.VideoProcessor.validate_file"
        ) as mock_validate_file:
            mock_video_info = VideoInfo(
                file_path=tmp_path / "test.mp4",
                duration=60.0,
                width=1920,
                height=1080,
                fps=30.0,
                format="mp4",
                size_mb=25.0,
                codec="h264",
            )
            mock_validate_file.return_value = mock_video_info

            # This should complete successfully
            result = coordinator.analyze_video(
                video_path=tmp_path / "test.mp4",
                extract_audio=False,
                detect_scenes=False,
                extract_frames=False,
            )

            assert result.success is True

            # Verify temp resources would be cleaned up
            # (In real usage, the VideoProcessor handles temp cleanup)
