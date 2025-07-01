"""Edge case tests for visual analysis components."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from deep_brief.analysis.frame_analyzer import FrameAnalysisPipeline
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
    VisualAnalysisResult,
)
from deep_brief.core.scene_detector import Scene as SceneInfo, SceneDetectionResult
from deep_brief.core.exceptions import ErrorCode, VideoProcessingError


class TestVisualAnalyzerEdgeCases:
    """Test edge cases in visual analyzer."""
    
    def test_extract_frames_with_save_disabled(self, tmp_path):
        """Test frame extraction with save_extracted_frames disabled."""
        config = Mock()
        config.visual_analysis.save_extracted_frames = False
        config.visual_analysis.frames_per_scene = 2
        config.visual_analysis.min_quality_score = 0.5
        config.visual_analysis.enable_quality_filtering = False
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        config.visual_analysis.blur_threshold = 100.0
        config.visual_analysis.contrast_threshold = 30.0
        config.visual_analysis.brightness_min = 20.0
        config.visual_analysis.brightness_max = 230.0
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=5.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=config)
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (True, np.ones((720, 1280, 3), dtype=np.uint8) * 128)
            
            result = extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=scene_result,
                output_dir=tmp_path / "output",  # Should be ignored
            )
            
            # Frames should not have file paths
            for scene in result.scene_analyses:
                for frame in scene.frames:
                    assert frame.file_path is None
    
    def test_extract_frames_with_save_and_resize(self, tmp_path):
        """Test frame extraction with saving and resizing."""
        config = Mock()
        config.visual_analysis.save_extracted_frames = True
        config.visual_analysis.frames_per_scene = 1
        config.visual_analysis.min_quality_score = 0.5
        config.visual_analysis.enable_quality_filtering = False
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        config.visual_analysis.blur_threshold = 100.0
        config.visual_analysis.contrast_threshold = 30.0
        config.visual_analysis.brightness_min = 20.0
        config.visual_analysis.brightness_max = 230.0
        config.visual_analysis.max_frame_width = 640
        config.visual_analysis.max_frame_height = 480
        config.visual_analysis.frame_quality = 90
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=2.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=config)
        
        # Large frame that needs resizing
        large_frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 200
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (True, large_frame)
            
            with patch('cv2.imwrite') as mock_imwrite:
                mock_imwrite.return_value = True
                
                result = extractor.extract_frames_from_scenes(
                    video_path=video_path,
                    scene_result=scene_result,
                    output_dir=output_dir,
                )
                
                # Verify imwrite was called
                mock_imwrite.assert_called()
                # Check that frame was resized
                saved_frame = mock_imwrite.call_args[0][1]
                assert saved_frame.shape[0] <= 480
                assert saved_frame.shape[1] <= 640
    
    def test_video_capture_release_on_error(self, tmp_path):
        """Test that video capture is released on error."""
        config = Mock()
        config.visual_analysis.frames_per_scene = 1
        config.visual_analysis.min_quality_score = 0.5
        config.visual_analysis.enable_quality_filtering = False
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=2.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=config)
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            # Simulate error during frame reading
            mock_cap.read.side_effect = Exception("Read error")
            
            with pytest.raises(VideoProcessingError):
                extractor.extract_frames_from_scenes(
                    video_path=video_path,
                    scene_result=scene_result,
                )
            
            # Verify release was called
            mock_cap.release.assert_called()
    
    def test_invalid_fps_fallback(self, tmp_path):
        """Test fallback when FPS is invalid."""
        config = Mock()
        config.visual_analysis.frames_per_scene = 1
        config.visual_analysis.min_quality_score = 0.5
        config.visual_analysis.enable_quality_filtering = False
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        config.visual_analysis.blur_threshold = 100.0
        config.visual_analysis.contrast_threshold = 30.0
        config.visual_analysis.brightness_min = 20.0
        config.visual_analysis.brightness_max = 230.0
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=1.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=config)
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 0.0  # Invalid FPS
            mock_cap.read.return_value = (True, np.ones((720, 1280, 3), dtype=np.uint8) * 128)
            
            # Should use fallback FPS
            result = extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=scene_result,
            )
            
            assert result.total_frames_extracted > 0
    
    def test_all_frames_filtered_by_quality(self, tmp_path):
        """Test when all frames are filtered out by quality."""
        config = Mock()
        config.visual_analysis.frames_per_scene = 3
        config.visual_analysis.min_quality_score = 0.95  # Very high threshold
        config.visual_analysis.enable_quality_filtering = True
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        config.visual_analysis.blur_threshold = 100.0
        config.visual_analysis.contrast_threshold = 30.0
        config.visual_analysis.brightness_min = 20.0
        config.visual_analysis.brightness_max = 230.0
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=3.0,
                    duration=3.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=3.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        extractor = FrameExtractor(config=config)
        
        # Low quality frame (uniform gray)
        low_quality_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (True, low_quality_frame)
            
            result = extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=scene_result,
            )
            
            # No frames should pass quality filter
            assert result.total_frames_extracted == 0
            assert result.total_frames_processed > 0
            assert result.overall_success_rate == 0.0


class TestFrameAnalyzerEdgeCases:
    """Test edge cases in frame analyzer."""
    
    def test_pipeline_with_no_enabled_analyses(self):
        """Test pipeline with all analyses disabled."""
        config = Mock()
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        
        pipeline = FrameAnalysisPipeline(config=config)
        
        # Only quality assessment should be enabled
        enabled = pipeline.get_enabled_analyses()
        assert enabled == ["quality_assessment"]
        
        # Total weight should only include quality assessment
        assert pipeline.total_weight == 0.2
    
    def test_analyze_single_frame_with_none_inputs(self):
        """Test single frame analysis with None frame info."""
        config = Mock()
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        
        pipeline = FrameAnalysisPipeline(config=config)
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Analyze with None frame_info
        result = pipeline.analyze_single_frame(frame, frame_info=None)
        
        assert isinstance(result, ExtractedFrame)
        assert result.frame_number == 0  # Default values
        assert result.timestamp == 0.0
        assert result.scene_number == 0
    
    def test_determine_content_type_edge_cases(self):
        """Test content type determination with edge cases."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.enable_object_detection = True
        
        pipeline = FrameAnalysisPipeline(config=config)
        
        # Empty visual result
        empty_result = VisualAnalysisResult(
            total_scenes=0,
            total_frames_extracted=0,
            total_frames_processed=0,
            overall_success_rate=0.0,
            scene_analyses=[],
            overall_quality_distribution={},
            average_quality_score=0.0,
            best_frames_per_scene=[],
            video_duration=0.0,
            extraction_method="scene_based",
            processing_time=0.0,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        from deep_brief.analysis.frame_analyzer import PipelineMetrics
        
        metrics = PipelineMetrics(
            total_frames_processed=0,
            successful_frames=0,
            failed_frames=0,
            total_processing_time=0.0,
        threshold_used=0.1,
        average_scene_duration=5.0,
            average_frame_time=0.0,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.0,
            quality_distribution={},
            frames_with_captions=0,
            frames_with_ocr=0,
            frames_with_objects=0,
        )
        
        content_type = pipeline._determine_content_type(empty_result, metrics)
        assert content_type == "unknown"
    
    def test_generate_summary_with_low_quality_video(self):
        """Test summary generation for low quality video."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.enable_object_detection = True
        
        pipeline = FrameAnalysisPipeline(config=config)
        
        # Create result with poor quality
        from deep_brief.analysis.visual_analyzer import SceneFrameAnalysis
        
        poor_quality_result = VisualAnalysisResult(
            total_scenes=2,
            total_frames_extracted=4,
            total_frames_processed=8,
            overall_success_rate=0.5,
            scene_analyses=[
                SceneFrameAnalysis(
                    scene_number=1,
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0,
                    frames=[],
                    total_frames_extracted=2,
                    best_frame=None,
                    average_quality_score=0.4,
                    quality_distribution={"poor": 2},
                    total_frames_processed=4,
                    frames_filtered_by_quality=2,
                    extraction_success_rate=0.5,
                ),
                SceneFrameAnalysis(
                    scene_number=2,
                    start_time=5.0,
                    end_time=10.0,
                    duration=5.0,
                    frames=[],
                    total_frames_extracted=2,
                    best_frame=None,
                    average_quality_score=0.45,
                    quality_distribution={"fair": 1, "poor": 1},
                    total_frames_processed=4,
                    frames_filtered_by_quality=2,
                    extraction_success_rate=0.5,
                ),
            ],
            overall_quality_distribution={"poor": 3, "fair": 1},
            average_quality_score=0.425,
            best_frames_per_scene=[],
            video_duration=10.0,
            extraction_method="scene_based",
            processing_time=2.0,
        )
        
        from deep_brief.analysis.frame_analyzer import PipelineMetrics
        
        metrics = PipelineMetrics(
            total_frames_processed=8,
            successful_frames=4,
            failed_frames=4,
            total_processing_time=2.0,
            average_frame_time=0.25,
            step_timings={
                "quality_assessment": 0.05,
                "image_captioning": 0.1,
                "text_extraction": 0.05,
                "object_detection": 0.05,
            },
            step_success_rates={
                "quality_assessment": 1.0,
                "image_captioning": 0.5,
                "text_extraction": 0.5,
                "object_detection": 0.5,
            },
            average_quality_score=0.425,
            quality_distribution={"poor": 3, "fair": 1},
            frames_with_captions=2,
            frames_with_ocr=2,
            frames_with_objects=2,
        )
        
        summary = pipeline.generate_analysis_summary(poor_quality_result, metrics)
        
        # Should identify quality issues
        assert "Overall video quality is below optimal levels" in summary["key_insights"]
        assert "More than half of the scenes have quality issues" in summary["key_insights"]
        assert summary["quality_summary"]["average_score"] < 0.6
    
    def test_progress_callback_with_invalid_update(self, tmp_path):
        """Test progress callback handling with invalid update format."""
        config = Mock()
        config.visual_analysis.enable_captioning = False
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.enable_object_detection = False
        
        pipeline = FrameAnalysisPipeline(config=config)
        
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        scene_result = SceneDetectionResult(
            scenes=[
                SceneInfo(
                    scene_number=1,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    confidence=0.9,
                    
                )
            ],
            total_scenes=1,
            video_duration=1.0,
            detection_method="threshold",
            processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
        )
        
        # Progress callback that expects specific format
        progress_values = []
        
        def progress_callback(progress: float, message: str):
            progress_values.append(progress)
        
        with patch.object(pipeline.frame_extractor, 'extract_frames_from_scenes') as mock_extract:
            mock_extract.return_value = VisualAnalysisResult(
                total_scenes=1,
                total_frames_extracted=1,
                total_frames_processed=1,
                overall_success_rate=1.0,
                scene_analyses=[],
                overall_quality_distribution={"good": 1},
                average_quality_score=0.8,
                best_frames_per_scene=[],
                video_duration=1.0,
                extraction_method="scene_based",
                processing_time=0.5,
        threshold_used=0.1,
        average_scene_duration=5.0,
            )
            
            # Should handle callback gracefully even with unexpected update format
            result, metrics = pipeline.analyze_video_frames(
                video_path=video_path,
                scene_result=scene_result,
                progress_callback=progress_callback,
            )
            
            assert result is not None
            assert metrics is not None


class TestComponentInitializationEdgeCases:
    """Test edge cases in component initialization."""
    
    def test_image_captioner_cleanup_without_models(self):
        """Test captioner cleanup when models were never loaded."""
        config = Mock()
        config.visual_analysis.enable_captioning = False
        
        captioner = ImageCaptioner(config=config)
        
        # Cleanup should work even if models were never loaded
        captioner.cleanup()
        
        assert captioner.model is None
        assert captioner.processor is None
    
    def test_ocr_detector_cleanup_without_initialization(self):
        """Test OCR detector cleanup without initialization."""
        config = Mock()
        config.visual_analysis.enable_ocr = False
        config.visual_analysis.ocr_engine = "tesseract"
        
        detector = OCRDetector(config=config)
        
        # Cleanup should work even if never initialized
        detector.cleanup()
    
    def test_object_detector_cleanup_without_model(self):
        """Test object detector cleanup without model loading."""
        config = Mock()
        config.visual_analysis.enable_object_detection = False
        
        detector = ObjectDetector(config=config)
        
        # Cleanup should work even if model was never loaded
        detector.cleanup()