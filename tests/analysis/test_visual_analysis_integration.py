"""Integration tests for visual analysis components."""

import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from deep_brief.analysis.error_handling import ImageValidationError
from deep_brief.analysis.frame_analyzer import (
    FrameAnalysisPipeline,
    FrameAnalysisStep,
    PipelineMetrics,
)
from deep_brief.analysis.image_captioner import CaptionResult, ImageCaptioner
from deep_brief.analysis.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    ObjectDetector,
    PresentationElement,
)
from deep_brief.analysis.ocr_detector import OCRDetector, OCRResult, TextRegion
from deep_brief.analysis.visual_analyzer import (
    ExtractedFrame,
    FrameExtractor,
    FrameQualityMetrics,
    SceneFrameAnalysis,
    VisualAnalysisResult,
)
from deep_brief.core.scene_detector import Scene as SceneInfo, SceneDetectionResult
from deep_brief.core.exceptions import ErrorCode, VideoProcessingError
from deep_brief.utils.config import get_config


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock()
    config.visual_analysis.enable_captioning = True
    config.visual_analysis.enable_ocr = True
    config.visual_analysis.enable_object_detection = True
    config.visual_analysis.frames_per_scene = 3
    config.visual_analysis.min_quality_score = 0.5
    config.visual_analysis.enable_quality_filtering = True
    config.visual_analysis.save_extracted_frames = False
    config.visual_analysis.blur_threshold = 100.0
    config.visual_analysis.contrast_threshold = 30.0
    config.visual_analysis.brightness_min = 20.0
    config.visual_analysis.brightness_max = 230.0
    config.visual_analysis.max_frame_width = 1920
    config.visual_analysis.max_frame_height = 1080
    config.visual_analysis.frame_quality = 90
    config.visual_analysis.captioning_model = "test-model"
    config.visual_analysis.ocr_engine = "tesseract"
    config.visual_analysis.ocr_confidence_threshold = 80.0
    config.visual_analysis.object_detection_confidence = 0.7
    return config


@pytest.fixture
def sample_scene_result():
    """Create sample scene detection result."""
    scenes = [
        SceneInfo(
            scene_number=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            confidence=0.95,
        ),
        SceneInfo(
            scene_number=2,
            start_time=5.0,
            end_time=10.0,
            duration=5.0,
            confidence=0.90,
        ),
    ]
    return SceneDetectionResult(
        scenes=scenes,
        total_scenes=2,
        video_duration=10.0,
        detection_method="threshold",
        processing_time=0.5,
        threshold_used=0.1,
        average_scene_duration=5.0,
    )


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    # Create a realistic-looking frame with text areas
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add a title area (dark text on light background)
    cv2.rectangle(frame, (100, 50), (1180, 150), (200, 200, 200), -1)
    cv2.putText(frame, "Sample Presentation Title", (150, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 3)
    
    # Add some content blocks
    cv2.rectangle(frame, (100, 200), (600, 400), (220, 220, 220), -1)
    cv2.rectangle(frame, (680, 200), (1180, 400), (220, 220, 220), -1)
    
    # Add a chart area
    cv2.rectangle(frame, (100, 450), (600, 650), (210, 210, 210), -1)
    # Draw some fake chart lines
    for i in range(5):
        y = 500 + i * 30
        cv2.line(frame, (150, y), (550, y), (150, 150, 150), 1)
    
    return frame


class TestVisualAnalysisIntegration:
    """Test integration between visual analysis components."""
    
    def test_full_visual_analysis_pipeline(self, mock_config, sample_scene_result, sample_frame, tmp_path):
        """Test complete visual analysis pipeline with all components."""
        # Create mock video file
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        # Create frame extractor with mocked components
        extractor = FrameExtractor(config=mock_config)
        
        # Mock the video capture
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, sample_frame)
            mock_cap.set.return_value = True
            
            # Mock the analysis components
            with patch.object(extractor, '_caption_frame') as mock_caption:
                mock_caption.return_value = CaptionResult(
                    caption="A presentation slide with title and charts",
                    confidence=0.85,
                    processing_time=0.5,
        threshold_used=0.1,
        average_scene_duration=5.0,
                    model_used="test-model",
                    tokens_generated=10,
                )
                
                with patch.object(extractor, '_extract_text_from_frame') as mock_ocr:
                    mock_ocr.return_value = OCRResult(
                        text_regions=[
                            TextRegion(
                                text="Sample Presentation Title",
                                confidence=95.0,
                                bbox=(100, 50, 1080, 100),
                                is_title=True,
                            )
                        ],
                        full_text="Sample Presentation Title",
                        processing_time=0.3,
        threshold_used=0.1,
        average_scene_duration=5.0,
                        engine_used="tesseract",
                        languages_detected=["en"],
                        total_text_regions=1,
                        high_confidence_regions=1,
                        average_confidence=95.0,
                    )
                    
                    with patch.object(extractor, '_detect_objects_in_frame') as mock_detect:
                        mock_detect.return_value = ObjectDetectionResult(
                            detected_objects=[
                                DetectedObject(
                                    element_type=PresentationElement.SLIDE,
                                    confidence=0.9,
                                    bbox=(0, 0, 1280, 720),
                                    center=(0.5, 0.5),
                                    area_ratio=1.0,
                                ),
                                DetectedObject(
                                    element_type=PresentationElement.TITLE,
                                    confidence=0.85,
                                    bbox=(100, 50, 1180, 150),
                                    center=(0.5, 0.14),
                                    area_ratio=0.1,
                                ),
                            ],
                            total_objects=2,
                            processing_time=0.2,
        threshold_used=0.1,
        average_scene_duration=5.0,
                            frame_width=1280,
                            frame_height=720,
                            element_counts={"slide": 1, "title": 1},
                            high_confidence_objects=2,
                            average_confidence=0.875,
                            layout_type="slide",
                            has_presenter=False,
                            dominant_element="slide",
                        )
                        
                        # Run the extraction
                        result = extractor.extract_frames_from_scenes(
                            video_path=video_path,
                            scene_result=sample_scene_result,
                            output_dir=None,
                        )
                        
                        # Verify the result
                        assert isinstance(result, VisualAnalysisResult)
                        assert result.total_scenes == 2
                        assert result.total_frames_extracted > 0
                        assert len(result.scene_analyses) == 2
                        
                        # Check first scene analysis
                        scene1 = result.scene_analyses[0]
                        assert scene1.scene_number == 1
                        assert len(scene1.frames) > 0
                        
                        # Check extracted frame
                        frame1 = scene1.frames[0]
                        assert isinstance(frame1, ExtractedFrame)
                        assert frame1.caption_result is not None
                        assert frame1.ocr_result is not None
                        assert frame1.object_detection_result is not None
                        
                        # Verify quality metrics
                        assert frame1.quality_metrics.overall_quality_score > 0
                        assert frame1.quality_metrics.blur_category in ["excellent", "good", "fair", "poor"]
    
    def test_error_recovery_in_pipeline(self, mock_config, sample_scene_result, tmp_path):
        """Test error recovery when components fail."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        extractor = FrameExtractor(config=mock_config)
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            
            # First frame is corrupt, second is valid
            mock_cap.read.side_effect = [
                (True, np.zeros((720, 1280, 3), dtype=np.uint8)),  # Black frame (corrupt)
                (True, np.ones((720, 1280, 3), dtype=np.uint8) * 128),  # Valid frame
                (False, None),  # End of video
            ]
            
            # Mock analysis components to fail on first call
            with patch.object(extractor, '_caption_frame') as mock_caption:
                mock_caption.side_effect = [
                    Exception("Caption model failed"),
                    CaptionResult(
                        caption="Recovery successful",
                        confidence=0.7,
                        processing_time=0.5,
        threshold_used=0.1,
        average_scene_duration=5.0,
                        model_used="test-model",
                        tokens_generated=5,
                    ),
                ]
                
                result = extractor.extract_frames_from_scenes(
                    video_path=video_path,
                    scene_result=sample_scene_result,
                    output_dir=None,
                )
                
                # Should still get results despite errors
                assert result is not None
                assert result.total_frames_processed >= 2
    
    def test_pipeline_with_progress_tracking(self, mock_config, sample_scene_result, sample_frame, tmp_path):
        """Test pipeline with progress callback."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        pipeline = FrameAnalysisPipeline(config=mock_config)
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(progress: float, message: str):
            progress_updates.append((progress, message))
        
        with patch.object(pipeline.frame_extractor, 'extract_frames_from_scenes') as mock_extract:
            # Create mock result
            mock_result = VisualAnalysisResult(
                total_scenes=2,
                total_frames_extracted=6,
                total_frames_processed=6,
                overall_success_rate=1.0,
                scene_analyses=[
                    SceneFrameAnalysis(
                        scene_number=1,
                        start_time=0.0,
                        end_time=5.0,
                        duration=5.0,
                        frames=[],
                        total_frames_extracted=3,
                        best_frame=None,
                        average_quality_score=0.8,
                        quality_distribution={"good": 3},
                        total_frames_processed=3,
                        frames_filtered_by_quality=0,
                        extraction_success_rate=1.0,
                    ),
                    SceneFrameAnalysis(
                        scene_number=2,
                        start_time=5.0,
                        end_time=10.0,
                        duration=5.0,
                        frames=[],
                        total_frames_extracted=3,
                        best_frame=None,
                        average_quality_score=0.75,
                        quality_distribution={"good": 2, "fair": 1},
                        total_frames_processed=3,
                        frames_filtered_by_quality=0,
                        extraction_success_rate=1.0,
                    ),
                ],
                overall_quality_distribution={"good": 5, "fair": 1},
                average_quality_score=0.775,
                best_frames_per_scene=[],
                video_duration=10.0,
                extraction_method="scene_based",
                processing_time=2.0,
            )
            mock_extract.return_value = mock_result
            
            # Run pipeline with progress tracking
            result, metrics = pipeline.analyze_video_frames(
                video_path=video_path,
                scene_result=sample_scene_result,
                output_dir=None,
                progress_callback=progress_callback,
            )
            
            assert result == mock_result
            assert isinstance(metrics, PipelineMetrics)
            assert len(progress_updates) > 0  # Should have received progress updates
    
    def test_frame_quality_filtering(self, mock_config, sample_scene_result):
        """Test quality-based frame filtering."""
        # Set strict quality threshold
        mock_config.visual_analysis.min_quality_score = 0.8
        mock_config.visual_analysis.enable_quality_filtering = True
        
        extractor = FrameExtractor(config=mock_config)
        
        # Create frames with varying quality
        high_quality_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        low_quality_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128  # Uniform gray (low contrast)
        
        # Test quality assessment
        high_metrics = extractor._assess_frame_quality(high_quality_frame)
        low_metrics = extractor._assess_frame_quality(low_quality_frame)
        
        # High quality frame should score better
        assert high_metrics.overall_quality_score > low_metrics.overall_quality_score
        assert high_metrics.contrast_score > low_metrics.contrast_score
    
    def test_content_type_detection(self, mock_config):
        """Test automatic content type detection."""
        pipeline = FrameAnalysisPipeline(config=mock_config)
        
        # Create visual result with slide-heavy content
        slide_result = VisualAnalysisResult(
            total_scenes=3,
            total_frames_extracted=9,
            total_frames_processed=9,
            overall_success_rate=1.0,
            scene_analyses=[
                self._create_scene_with_layout("slide", 3),
                self._create_scene_with_layout("slide", 3),
                self._create_scene_with_layout("video", 3),
            ],
            overall_quality_distribution={"good": 9},
            average_quality_score=0.8,
            best_frames_per_scene=[],
            video_duration=15.0,
            extraction_method="scene_based",
            processing_time=3.0,
        )
        
        metrics = PipelineMetrics(
            total_frames_processed=9,
            successful_frames=9,
            failed_frames=0,
            total_processing_time=3.0,
            average_frame_time=0.33,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.8,
            quality_distribution={"good": 9},
            frames_with_captions=9,
            frames_with_ocr=9,
            frames_with_objects=9,
        )
        
        summary = pipeline.generate_analysis_summary(slide_result, metrics)
        assert summary["content_analysis"]["content_type"] == "presentation"
    
    def test_error_handling_with_corrupt_video(self, mock_config, sample_scene_result, tmp_path):
        """Test handling of corrupt video files."""
        video_path = tmp_path / "corrupt_video.mp4"
        video_path.touch()
        
        extractor = FrameExtractor(config=mock_config)
        
        # Mock video that fails to open
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = False
            
            with pytest.raises(VideoProcessingError) as exc_info:
                extractor.extract_frames_from_scenes(
                    video_path=video_path,
                    scene_result=sample_scene_result,
                )
            
            assert exc_info.value.error_code == ErrorCode.VIDEO_READ_ERROR
    
    def test_performance_metrics_collection(self, mock_config, sample_scene_result, sample_frame, tmp_path):
        """Test collection of performance metrics."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        pipeline = FrameAnalysisPipeline(config=mock_config)
        
        with patch.object(pipeline.frame_extractor, 'extract_frames_from_scenes') as mock_extract:
            # Create result with timing information
            mock_result = self._create_mock_visual_result()
            mock_extract.return_value = mock_result
            
            start_time = time.time()
            result, metrics = pipeline.analyze_video_frames(
                video_path=video_path,
                scene_result=sample_scene_result,
            )
            
            # Check metrics
            assert metrics.total_processing_time > 0
            assert metrics.average_frame_time > 0
            assert metrics.success_rate == 1.0
            assert "quality_assessment" in metrics.step_timings
            assert metrics.frames_with_captions == 0  # No real captions in mock
    
    def _create_scene_with_layout(self, layout_type: str, num_frames: int) -> SceneFrameAnalysis:
        """Helper to create scene analysis with specific layout type."""
        frames = []
        for i in range(num_frames):
            frames.append(
                ExtractedFrame(
                    frame_number=i * 30,
                    timestamp=float(i),
                    scene_number=1,
                    width=1280,
                    height=720,
                    quality_metrics=FrameQualityMetrics(
                        blur_score=150.0,
                        blur_category="good",
                        contrast_score=40.0,
                        contrast_category="good",
                        brightness_score=128.0,
                        brightness_category="good",
                        overall_quality_score=0.8,
                        overall_quality_category="good",
                        histogram_metrics={},
                        sharpness_details={},
                        color_metrics={},
                        noise_metrics={},
                        composition_metrics={},
                        quality_report={},
                    ),
                    object_detection_result=ObjectDetectionResult(
                        detected_objects=[
                            DetectedObject(
                                element_type=PresentationElement.SLIDE,
                                confidence=0.9,
                                bbox=(0, 0, 1280, 720),
                                center=(0.5, 0.5),
                                area_ratio=1.0,
                            )
                        ],
                        total_objects=1,
                        processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
                        frame_width=1280,
                        frame_height=720,
                        element_counts={"slide": 1},
                        high_confidence_objects=1,
                        average_confidence=0.9,
                        layout_type=layout_type,
                        has_presenter=False,
                        dominant_element="slide",
                    ),
                )
            )
        
        return SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            frames=frames,
            total_frames_extracted=num_frames,
            best_frame=frames[0] if frames else None,
            average_quality_score=0.8,
            quality_distribution={"good": num_frames},
            total_frames_processed=num_frames,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )
    
    def _create_mock_visual_result(self) -> VisualAnalysisResult:
        """Helper to create a mock visual analysis result."""
        return VisualAnalysisResult(
            total_scenes=1,
            total_frames_extracted=3,
            total_frames_processed=3,
            overall_success_rate=1.0,
            scene_analyses=[self._create_scene_with_layout("slide", 3)],
            overall_quality_distribution={"good": 3},
            average_quality_score=0.8,
            best_frames_per_scene=[],
            video_duration=5.0,
            extraction_method="scene_based",
            processing_time=1.0,
        )


class TestComponentCleanup:
    """Test cleanup of visual analysis components."""
    
    def test_frame_extractor_cleanup(self, mock_config):
        """Test cleanup of frame extractor resources."""
        extractor = FrameExtractor(config=mock_config)
        
        # Create mock components
        extractor.captioner = Mock()
        extractor.ocr_detector = Mock()
        extractor.object_detector = Mock()
        
        # Run cleanup
        extractor.cleanup()
        
        # Verify cleanup was called on all components
        extractor.captioner.cleanup.assert_called_once()
        extractor.ocr_detector.cleanup.assert_called_once()
        extractor.object_detector.cleanup.assert_called_once()
        
        # Verify components were set to None
        assert extractor.captioner is None
        assert extractor.ocr_detector is None
        assert extractor.object_detector is None
    
    def test_pipeline_cleanup(self, mock_config):
        """Test cleanup of pipeline resources."""
        pipeline = FrameAnalysisPipeline(config=mock_config)
        
        # Mock the frame extractor
        pipeline.frame_extractor = Mock()
        
        # Run cleanup
        pipeline.cleanup()
        
        # Verify cleanup was called
        pipeline.frame_extractor.cleanup.assert_called_once()


class TestEdgeCases:
    """Test edge cases in visual analysis."""
    
    def test_empty_scene_result(self, mock_config, tmp_path):
        """Test handling of empty scene detection result."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        empty_scene_result = SceneDetectionResult(
            scenes=[],
            total_scenes=0,
            video_duration=10.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=mock_config)
        
        with pytest.raises(VideoProcessingError) as exc_info:
            extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=empty_scene_result,
            )
        
        assert "No scenes detected" in str(exc_info.value)
    
    def test_single_frame_scene(self, mock_config, sample_frame, tmp_path):
        """Test extraction from scene with single frame."""
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()
        
        # Scene with very short duration
        single_frame_scene = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=0.033,
                    duration=0.033,  # Single frame at 30fps
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=0.033,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=mock_config)
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (True, sample_frame)
            
            result = extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=single_frame_scene,
            )
            
            # Should extract at least one frame
            assert result.total_frames_extracted >= 1
    
    def test_invalid_frame_dimensions(self, mock_config):
        """Test handling of frames with invalid dimensions."""
        extractor = FrameExtractor(config=mock_config)
        
        # Frame too small
        tiny_frame = np.ones((5, 5, 3), dtype=np.uint8)
        
        # Quality assessment should handle small frames
        metrics = extractor._assess_frame_quality(tiny_frame)
        assert metrics is not None
        assert metrics.overall_quality_score >= 0