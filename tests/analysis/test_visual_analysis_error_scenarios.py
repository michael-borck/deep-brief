"""Error scenario tests for visual analysis components."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from deep_brief.analysis.error_handling import (
    ImageValidationError,
    ModelInitializationError,
)
from deep_brief.analysis.frame_analyzer import FrameAnalysisPipeline
from deep_brief.analysis.image_captioner import CaptionResult, ImageCaptioner
from deep_brief.analysis.object_detector import ObjectDetector
from deep_brief.analysis.ocr_detector import OCRDetector
from deep_brief.analysis.visual_analyzer import FrameExtractor
from deep_brief.core.exceptions import ErrorCode, VideoProcessingError
from deep_brief.core.scene_detector import Scene as SceneInfo, SceneDetectionResult


class TestImageCaptionerErrors:
    """Test error handling in image captioner."""
    
    def test_model_loading_failure(self):
        """Test handling of model loading failures."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.captioning_model = "invalid-model-name"
        config.visual_analysis.captioning_device = "auto"
        
        captioner = ImageCaptioner(config=config)
        
        # Mock transformers to fail
        with patch('deep_brief.analysis.image_captioner.Blip2Processor.from_pretrained') as mock_processor:
            mock_processor.side_effect = Exception("Model not found")
            
            with pytest.raises(ModelInitializationError) as exc_info:
                captioner._load_model()
            
            assert exc_info.value.error_code == ErrorCode.MODEL_LOADING_FAILED
            assert "invalid-model-name" in exc_info.value.details["model_name"]
    
    def test_caption_with_cuda_oom(self):
        """Test handling of CUDA out of memory errors."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.captioning_model = "test-model"
        config.visual_analysis.captioning_device = "cuda"
        config.visual_analysis.max_caption_length = 50
        config.visual_analysis.caption_temperature = 1.0
        
        captioner = ImageCaptioner(config=config)
        
        # Mock model and processor
        captioner.model = Mock()
        captioner.processor = Mock()
        captioner.device = "cuda"
        
        # Mock CUDA OOM error
        captioner.model.generate.side_effect = RuntimeError("CUDA out of memory")
        captioner.processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        captioner.processor.tokenizer = Mock(pad_token_id=0)
        
        # Should raise with proper error code
        with pytest.raises(VideoProcessingError) as exc_info:
            captioner.caption_image(pil_image=Image.new("RGB", (224, 224)))
        
        assert exc_info.value.error_code == ErrorCode.INSUFFICIENT_MEMORY
    
    def test_caption_with_missing_dependency(self):
        """Test handling of missing dependencies."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.captioning_model = "test-model"
        config.visual_analysis.captioning_device = "cpu"
        config.visual_analysis.max_caption_length = 50
        config.visual_analysis.caption_temperature = 1.0
        
        captioner = ImageCaptioner(config=config)
        
        # Mock model loading to simulate import error
        with patch('deep_brief.analysis.image_captioner.Blip2Processor') as mock_processor:
            mock_processor.side_effect = ImportError("No module named 'transformers'")
            
            # Should raise with proper error code
            with pytest.raises(VideoProcessingError) as exc_info:
                captioner.caption_image(pil_image=Image.new("RGB", (224, 224)))
            
            assert exc_info.value.error_code == ErrorCode.MISSING_DEPENDENCY
    
    def test_caption_batch_with_mixed_failures(self):
        """Test batch captioning with some failures."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.captioning_model = "test-model"
        config.visual_analysis.captioning_device = "cpu"
        config.visual_analysis.caption_batch_size = 2
        
        captioner = ImageCaptioner(config=config)
        
        # Create test images
        images = [
            Path("/nonexistent/image1.jpg"),  # Will fail
            Image.new("RGB", (224, 224)),      # Should succeed
            Path("/nonexistent/image2.jpg"),  # Will fail
        ]
        
        # Mock successful caption for PIL image
        with patch.object(captioner, 'caption_image') as mock_caption:
            mock_caption.side_effect = [
                VideoProcessingError("File not found", ErrorCode.FILE_NOT_FOUND),
                CaptionResult(
                    caption="Test caption",
                    confidence=0.8,
                    processing_time=0.1,
        threshold_used=0.1,
        average_scene_duration=5.0,
                    model_used="test-model",
                    tokens_generated=5,
                ),
                VideoProcessingError("File not found", ErrorCode.FILE_NOT_FOUND),
            ]
            
            results = captioner.caption_batch(images)
            
            assert len(results) == 3
            assert "Caption failed" in results[0].caption
            assert results[1].caption == "Test caption"
            assert "Caption failed" in results[2].caption


class TestOCRDetectorErrors:
    """Test error handling in OCR detector."""
    
    def test_tesseract_not_installed(self):
        """Test handling when Tesseract is not installed."""
        config = Mock()
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.ocr_engine = "tesseract"
        config.visual_analysis.ocr_languages = ["eng"]
        config.visual_analysis.ocr_confidence_threshold = 80.0
        config.visual_analysis.ocr_text_min_length = 3
        
        with patch('deep_brief.analysis.ocr_detector.TESSERACT_AVAILABLE', False):
            with pytest.raises(VideoProcessingError) as exc_info:
                OCRDetector(config=config)
            
            assert exc_info.value.error_code == ErrorCode.MISSING_DEPENDENCY
            assert "pytesseract" in str(exc_info.value)
    
    def test_easyocr_initialization_failure(self):
        """Test handling of EasyOCR initialization failure."""
        config = Mock()
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.ocr_engine = "easyocr"
        config.visual_analysis.ocr_languages = ["en", "es"]
        config.visual_analysis.ocr_confidence_threshold = 80.0
        config.visual_analysis.ocr_text_min_length = 3
        
        with patch('deep_brief.analysis.ocr_detector.EASYOCR_AVAILABLE', True):
            with patch('deep_brief.analysis.ocr_detector.easyocr.Reader') as mock_reader:
                mock_reader.side_effect = Exception("Failed to download model")
                
                with pytest.raises(ModelInitializationError) as exc_info:
                    OCRDetector(config=config)
                
                assert exc_info.value.error_code == ErrorCode.MODEL_LOADING_FAILED
                assert "easyocr" in exc_info.value.details["model_name"]
    
    def test_ocr_with_corrupt_image(self):
        """Test OCR with corrupt image input."""
        config = Mock()
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.ocr_engine = "tesseract"
        config.visual_analysis.ocr_languages = ["eng"]
        config.visual_analysis.ocr_confidence_threshold = 80.0
        config.visual_analysis.ocr_text_min_length = 3
        
        detector = OCRDetector(config=config)
        detector.backend_initialized = True
        
        # Try to process invalid image
        with pytest.raises(ImageValidationError):
            detector.detect_text(image_array=np.array([1, 2, 3]))  # 1D array


class TestObjectDetectorErrors:
    """Test error handling in object detector."""
    
    def test_torch_not_available(self):
        """Test handling when PyTorch is not available."""
        config = Mock()
        config.visual_analysis.enable_object_detection = True
        config.visual_analysis.object_detection_device = "auto"
        config.visual_analysis.object_detection_confidence = 0.7
        
        detector = ObjectDetector(config=config)
        
        with patch('deep_brief.analysis.object_detector.torch') as mock_torch:
            mock_torch.side_effect = ImportError("No module named 'torch'")
            
            with pytest.raises(ModelInitializationError):
                detector._load_model()
    
    def test_detect_with_invalid_image_path(self):
        """Test detection with invalid image path."""
        config = Mock()
        config.visual_analysis.enable_object_detection = True
        config.visual_analysis.object_detection_device = "cpu"
        config.visual_analysis.object_detection_confidence = 0.7
        
        detector = ObjectDetector(config=config)
        detector._initialized = True
        
        # Should return empty result for nonexistent file
        result = detector.detect_objects(image_path=Path("/nonexistent/image.jpg"))
        
        assert result.total_objects == 0
        assert result.layout_type == "unknown"


class TestFrameExtractorErrors:
    """Test error handling in frame extractor."""
    
    def test_corrupt_frame_handling(self, tmp_path):
        """Test handling of corrupt frames during extraction."""
        config = Mock()
        config.visual_analysis.frames_per_scene = 3
        config.visual_analysis.min_quality_score = 0.5
        config.visual_analysis.enable_quality_filtering = False
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.enable_object_detection = True
        config.visual_analysis.save_extracted_frames = False
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
        
        # Create frames: black (corrupt), normal, white (corrupt)
        frames = [
            np.zeros((720, 1280, 3), dtype=np.uint8),  # Black
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),  # Normal
            np.full((720, 1280, 3), 255, dtype=np.uint8),  # White
        ]
        
        frame_index = 0
        def read_frame():
            nonlocal frame_index
            if frame_index < len(frames):
                frame = frames[frame_index]
                frame_index += 1
                return True, frame
            return False, None
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.side_effect = read_frame
            mock_cap.set.return_value = True
            
            # Mock analysis components to track calls
            with patch.object(extractor, '_caption_frame') as mock_caption:
                mock_caption.return_value = None
                
                with patch.object(extractor, '_extract_text_from_frame') as mock_ocr:
                    mock_ocr.return_value = None
                    
                    with patch.object(extractor, '_detect_objects_in_frame') as mock_detect:
                        mock_detect.return_value = None
                        
                        result = extractor.extract_frames_from_scenes(
                            video_path=video_path,
                            scene_result=scene_result,
                        )
                        
                        # Should process fewer frames due to corrupt ones
                        assert result.total_frames_processed == 3
                        # Only the normal frame should be extracted
                        assert result.total_frames_extracted == 1
    
    def test_scene_with_all_read_failures(self, tmp_path):
        """Test scene where all frame reads fail."""
        config = Mock()
        config.visual_analysis.frames_per_scene = 3
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
        
        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (False, None)  # Always fail
            
            result = extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=scene_result,
            )
            
            # Should handle gracefully with no extracted frames
            assert result.total_frames_extracted == 0
            assert result.total_frames_processed == 3
            assert result.overall_success_rate == 0.0


class TestPipelineErrorPropagation:
    """Test error propagation through the pipeline."""
    
    def test_pipeline_continues_with_component_failures(self, tmp_path):
        """Test that pipeline continues when individual components fail."""
        config = Mock()
        config.visual_analysis.enable_captioning = True
        config.visual_analysis.enable_ocr = True
        config.visual_analysis.enable_object_detection = True
        
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
        
        # Mock frame extractor to simulate component failures
        with patch.object(pipeline.frame_extractor, '_caption_frame') as mock_caption:
            mock_caption.side_effect = Exception("Caption model crashed")
            
            with patch.object(pipeline.frame_extractor, '_extract_text_from_frame') as mock_ocr:
                mock_ocr.side_effect = Exception("OCR engine crashed")
                
                with patch.object(pipeline.frame_extractor, '_detect_objects_in_frame') as mock_detect:
                    mock_detect.side_effect = Exception("Object detector crashed")
                    
                    # Mock the main extraction to return a result
                    with patch.object(pipeline.frame_extractor, 'extract_frames_from_scenes') as mock_extract:
                        from deep_brief.analysis.visual_analyzer import VisualAnalysisResult
                        
                        mock_extract.return_value = VisualAnalysisResult(
                            total_scenes=1,
                            total_frames_extracted=1,
                            total_frames_processed=1,
                            overall_success_rate=1.0,
                            scene_analyses=[],
                            overall_quality_distribution={"fair": 1},
                            average_quality_score=0.6,
                            best_frames_per_scene=[],
                            video_duration=1.0,
                            extraction_method="scene_based",
                            processing_time=0.5,
        threshold_used=0.1,
        average_scene_duration=5.0,
                        )
                        
                        # Pipeline should complete despite component failures
                        result, metrics = pipeline.analyze_video_frames(
                            video_path=video_path,
                            scene_result=scene_result,
                        )
                        
                        assert result is not None
                        assert metrics is not None
                        assert metrics.frames_with_captions == 0
                        assert metrics.frames_with_ocr == 0
                        assert metrics.frames_with_objects == 0