"""Tests for image captioning functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from deep_brief.analysis.image_captioner import (
    CaptionResult,
    ImageCaptioner,
    create_image_captioner,
)
from deep_brief.core.exceptions import ErrorCode, VideoProcessingError
from deep_brief.utils.config import DeepBriefConfig, VisualAnalysisConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = DeepBriefConfig(
        visual_analysis=VisualAnalysisConfig(
            enable_captioning=True,
            captioning_model="Salesforce/blip2-opt-2.7b",
            captioning_device="cpu",
            max_caption_length=50,
            caption_temperature=1.0,
            caption_batch_size=2,
        )
    )
    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock configuration with captioning disabled."""
    config = DeepBriefConfig(
        visual_analysis=VisualAnalysisConfig(
            enable_captioning=False,
            captioning_model="Salesforce/blip2-opt-2.7b",
            captioning_device="cpu",
        )
    )
    return config


@pytest.fixture
def image_captioner(mock_config):
    """Create ImageCaptioner instance for testing."""
    return ImageCaptioner(config=mock_config)


@pytest.fixture
def sample_image():
    """Create sample PIL Image for testing."""
    # Create a simple test image (100x100 RGB)
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add some pattern to make it interesting
    image_array[25:75, 25:75] = [128, 128, 128]  # Gray square
    image_array[40:60, 40:60] = [255, 255, 255]  # White square in center

    return Image.fromarray(image_array)


@pytest.fixture
def sample_image_array():
    """Create sample numpy array for testing."""
    # Create a simple test image (100x100 RGB)
    image_array = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add some pattern to make it interesting
    image_array[25:75, 25:75] = [64, 128, 192]  # Colored square
    image_array[40:60, 40:60] = [255, 0, 0]  # Red square in center

    return image_array


class TestCaptionResult:
    """Test CaptionResult model."""

    def test_caption_result_creation(self):
        """Test creating CaptionResult object."""
        result = CaptionResult(
            caption="A person standing in a room",
            confidence=0.85,
            processing_time=2.5,
            model_used="Salesforce/blip2-opt-2.7b",
            tokens_generated=12,
            alternative_captions=["Someone in a room", "A figure in an indoor space"],
        )

        assert result.caption == "A person standing in a room"
        assert result.confidence == 0.85
        assert result.processing_time == 2.5
        assert result.model_used == "Salesforce/blip2-opt-2.7b"
        assert result.tokens_generated == 12
        assert len(result.alternative_captions) == 2


class TestImageCaptioner:
    """Test ImageCaptioner class."""

    def test_initialization_default_config(self):
        """Test ImageCaptioner initialization with default config."""
        with patch("deep_brief.analysis.image_captioner.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.visual_analysis.captioning_device = "cpu"
            mock_get_config.return_value = mock_config

            captioner = ImageCaptioner()

            assert captioner.config == mock_config
            assert captioner.device == "cpu"
            mock_get_config.assert_called_once()

    def test_initialization_custom_config(self, mock_config):
        """Test ImageCaptioner initialization with custom config."""
        captioner = ImageCaptioner(config=mock_config)

        assert captioner.config == mock_config
        assert captioner.device == "cpu"

    def test_determine_device_auto_cpu(self, mock_config):
        """Test device determination with auto setting (CPU fallback)."""
        mock_config.visual_analysis.captioning_device = "auto"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            captioner = ImageCaptioner(config=mock_config)
            assert captioner.device == "cpu"

    def test_determine_device_auto_cuda(self, mock_config):
        """Test device determination with auto setting (CUDA available)."""
        mock_config.visual_analysis.captioning_device = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            captioner = ImageCaptioner(config=mock_config)
            assert captioner.device == "cuda"

    def test_determine_device_auto_mps(self, mock_config):
        """Test device determination with auto setting (MPS available)."""
        mock_config.visual_analysis.captioning_device = "auto"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            captioner = ImageCaptioner(config=mock_config)
            assert captioner.device == "mps"

    def test_determine_device_explicit(self, mock_config):
        """Test device determination with explicit setting."""
        mock_config.visual_analysis.captioning_device = "cuda"
        captioner = ImageCaptioner(config=mock_config)
        assert captioner.device == "cuda"

    @patch("deep_brief.analysis.image_captioner.Blip2Processor")
    @patch("deep_brief.analysis.image_captioner.Blip2ForConditionalGeneration")
    def test_load_model_blip2(
        self, mock_model_cls, mock_processor_cls, image_captioner
    ):
        """Test loading BLIP-2 model."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock the .to() method to return the same mock object
        mock_model.to.return_value = mock_model

        model, processor = image_captioner._load_model()

        assert model == mock_model
        assert processor == mock_processor
        mock_processor_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    @patch("deep_brief.analysis.image_captioner.BlipProcessor")
    @patch("deep_brief.analysis.image_captioner.BlipForConditionalGeneration")
    def test_load_model_blip(self, mock_model_cls, mock_processor_cls, mock_config):
        """Test loading original BLIP model."""
        mock_config.visual_analysis.captioning_model = (
            "Salesforce/blip-image-captioning-base"
        )
        captioner = ImageCaptioner(config=mock_config)

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock the .to() method to return the same mock object
        mock_model.to.return_value = mock_model

        model, processor = captioner._load_model()

        assert model == mock_model
        assert processor == mock_processor
        mock_processor_cls.from_pretrained.assert_called_once()
        mock_model_cls.from_pretrained.assert_called_once()

    @patch("deep_brief.analysis.image_captioner.Blip2Processor")
    @patch("deep_brief.analysis.image_captioner.Blip2ForConditionalGeneration")
    def test_load_model_failure(
        self, mock_model_cls, mock_processor_cls, image_captioner
    ):
        """Test model loading failure."""
        mock_processor_cls.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(VideoProcessingError) as exc_info:
            image_captioner._load_model()

        assert exc_info.value.error_code == ErrorCode.MISSING_DEPENDENCY
        assert "Failed to load image captioning model" in str(exc_info.value)

    def test_caption_image_disabled(self, mock_config_disabled):
        """Test image captioning when disabled."""
        captioner = ImageCaptioner(config=mock_config_disabled)

        result = captioner.caption_image(pil_image=Image.new("RGB", (100, 100)))

        assert result.caption == "Image captioning disabled"
        assert result.confidence == 0.0
        assert result.model_used == "none"

    def test_caption_image_no_input(self, image_captioner):
        """Test image captioning with no input provided."""
        with pytest.raises(ValueError) as exc_info:
            image_captioner.caption_image()

        assert "Must provide one of" in str(exc_info.value)

    def test_caption_image_pil_success(self, image_captioner, sample_image):
        """Test successful image captioning with PIL Image."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processing and generation - need a mock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Chain the .to() call
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs
        mock_processor.batch_decode.return_value = [
            "A test image with geometric shapes"
        ]

        # Patch the _load_model method to return our mocks
        with patch.object(
            image_captioner, "_load_model", return_value=(mock_model, mock_processor)
        ):
            result = image_captioner.caption_image(pil_image=sample_image)

        assert isinstance(result, CaptionResult)
        assert result.caption == "A test image with geometric shapes"
        assert result.confidence > 0
        assert result.processing_time > 0
        assert result.model_used == "Salesforce/blip2-opt-2.7b"
        assert result.tokens_generated > 0

    def test_caption_image_array_success(self, image_captioner, sample_image_array):
        """Test successful image captioning with numpy array."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processing and generation - need a mock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Chain the .to() call
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs
        mock_processor.batch_decode.return_value = [
            "An image with colorful geometric patterns"
        ]

        # Patch the _load_model method to return our mocks
        with patch.object(
            image_captioner, "_load_model", return_value=(mock_model, mock_processor)
        ):
            result = image_captioner.caption_image(image_array=sample_image_array)

        assert isinstance(result, CaptionResult)
        assert result.caption == "An image with colorful geometric patterns"
        assert result.confidence > 0
        assert result.processing_time > 0

    def test_caption_image_path_not_found(self, image_captioner):
        """Test image captioning with non-existent file."""
        non_existent_path = Path("/nonexistent/image.jpg")

        with pytest.raises(VideoProcessingError) as exc_info:
            image_captioner.caption_image(image_path=non_existent_path)

        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND
        assert "Image file not found" in str(exc_info.value)

    def test_caption_image_path_success(self, image_captioner, sample_image):
        """Test successful image captioning with file path."""
        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            sample_image.save(temp_file.name)
            image_path = Path(temp_file.name)

        try:
            # Mock model and processor
            mock_processor = MagicMock()
            mock_model = MagicMock()

            # Mock processing and generation
            mock_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
            mock_processor.return_value = mock_inputs
            mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model.generate.return_value = mock_outputs
            mock_processor.batch_decode.return_value = ["A saved image with patterns"]

            # Patch the _load_model method to return our mocks
            with patch.object(
                image_captioner,
                "_load_model",
                return_value=(mock_model, mock_processor),
            ):
                result = image_captioner.caption_image(image_path=image_path)

            assert isinstance(result, CaptionResult)
            assert result.caption == "A saved image with patterns"
            assert result.confidence > 0

        finally:
            # Clean up
            if image_path.exists():
                image_path.unlink()

    def test_caption_image_with_prompt(self, image_captioner, sample_image):
        """Test image captioning with custom prompt."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processing and generation - need a mock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Chain the .to() call
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs
        mock_processor.batch_decode.return_value = [
            "A presentation slide with geometric elements"
        ]

        # Patch the _load_model method to return our mocks
        with patch.object(
            image_captioner, "_load_model", return_value=(mock_model, mock_processor)
        ):
            result = image_captioner.caption_image(
                pil_image=sample_image, prompt="Describe this presentation slide"
            )

        assert isinstance(result, CaptionResult)
        assert result.caption == "A presentation slide with geometric elements"
        # Verify processor was called with both image and prompt
        # We need to check the call arguments since mock_processor() was called
        call_args = mock_processor.call_args
        assert call_args[0][0] == sample_image  # First argument is the image
        assert (
            call_args[0][1] == "Describe this presentation slide"
        )  # Second argument is prompt
        assert call_args[1]["return_tensors"] == "pt"  # Keyword argument

    @patch("deep_brief.analysis.image_captioner.Blip2Processor")
    @patch("deep_brief.analysis.image_captioner.Blip2ForConditionalGeneration")
    def test_caption_image_generation_failure(
        self, mock_model_cls, mock_processor_cls, image_captioner, sample_image
    ):
        """Test image captioning with generation failure."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock processing but make generation fail
        mock_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = mock_inputs
        mock_model.generate.side_effect = Exception("CUDA out of memory")

        with pytest.raises(VideoProcessingError) as exc_info:
            image_captioner.caption_image(pil_image=sample_image)

        assert exc_info.value.error_code == ErrorCode.INSUFFICIENT_MEMORY
        assert "Image captioning failed" in str(exc_info.value)

    def test_estimate_caption_confidence(self, image_captioner):
        """Test caption confidence estimation heuristics."""
        # Test different caption types
        good_caption = "A person standing in a conference room presenting slides"
        short_caption = "Room"
        empty_caption = ""
        failed_caption = "Unable to generate caption"
        verbose_caption = "This is a very long and detailed description of an image that goes on and on with many unnecessary words and details that make it too verbose"

        assert image_captioner._estimate_caption_confidence(good_caption) > 0.6
        assert image_captioner._estimate_caption_confidence(short_caption) < 0.4
        assert image_captioner._estimate_caption_confidence(empty_caption) == 0.0
        assert image_captioner._estimate_caption_confidence(failed_caption) == 0.0
        assert image_captioner._estimate_caption_confidence(verbose_caption) < 0.7

    def test_caption_batch_success(self, image_captioner, sample_image):
        """Test batch captioning functionality."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processing and generation - need a mock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Chain the .to() call
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs

        # Return different captions for each call
        mock_processor.batch_decode.side_effect = [
            ["First image caption"],
            ["Second image caption"],
            ["Third image caption"],
        ]

        images = [sample_image, sample_image, sample_image]

        # Patch the _load_model method to return our mocks
        with patch.object(
            image_captioner, "_load_model", return_value=(mock_model, mock_processor)
        ):
            results = image_captioner.caption_batch(images)

        assert len(results) == 3
        assert results[0].caption == "First image caption"
        assert results[1].caption == "Second image caption"
        assert results[2].caption == "Third image caption"

    @patch("deep_brief.analysis.image_captioner.Blip2Processor")
    @patch("deep_brief.analysis.image_captioner.Blip2ForConditionalGeneration")
    def test_caption_batch_with_prompts(
        self, mock_model_cls, mock_processor_cls, image_captioner, sample_image
    ):
        """Test batch captioning with prompts."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock processing and generation - need a mock with .to() method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # Chain the .to() call
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_outputs
        mock_processor.batch_decode.side_effect = [
            ["A presentation slide"],
            ["A conference room"],
        ]

        images = [sample_image, sample_image]
        prompts = ["Describe this slide", "Describe this room"]
        results = image_captioner.caption_batch(images, prompts=prompts)

        assert len(results) == 2
        assert results[0].caption == "A presentation slide"
        assert results[1].caption == "A conference room"

    def test_caption_batch_prompt_mismatch(self, image_captioner, sample_image):
        """Test batch captioning with mismatched prompts."""
        images = [sample_image, sample_image]
        prompts = ["Only one prompt"]

        with pytest.raises(ValueError) as exc_info:
            image_captioner.caption_batch(images, prompts=prompts)

        assert "Number of prompts must match number of images" in str(exc_info.value)

    @patch("deep_brief.analysis.image_captioner.Blip2Processor")
    @patch("deep_brief.analysis.image_captioner.Blip2ForConditionalGeneration")
    def test_caption_batch_partial_failure(
        self, mock_model_cls, mock_processor_cls, image_captioner, sample_image
    ):
        """Test batch captioning with some failures."""
        # Mock model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock processing and generation - first succeeds, second fails
        mock_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_processor.return_value = mock_inputs
        mock_outputs = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.side_effect = [mock_outputs, Exception("Model error")]
        mock_processor.batch_decode.return_value = ["Successful caption"]

        images = [sample_image, sample_image]
        results = image_captioner.caption_batch(images)

        assert len(results) == 2
        assert results[0].caption == "Successful caption"
        assert "Caption failed" in results[1].caption
        assert results[1].confidence == 0.0

    def test_get_supported_models(self, image_captioner):
        """Test getting supported model list."""
        models = image_captioner.get_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "Salesforce/blip2-opt-2.7b" in models
        assert "Salesforce/blip-image-captioning-base" in models

    def test_cleanup(self, image_captioner):
        """Test model cleanup functionality."""
        # Set up some mock models
        image_captioner.model = MagicMock()
        image_captioner.processor = MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
        ):
            image_captioner.cleanup()

            assert image_captioner.model is None
            assert image_captioner.processor is None
            mock_empty_cache.assert_called_once()


class TestImageCaptionerFactory:
    """Test ImageCaptioner factory function."""

    def test_create_image_captioner_no_config(self):
        """Test creating ImageCaptioner without config."""
        with patch("deep_brief.analysis.image_captioner.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.visual_analysis.captioning_device = "cpu"
            mock_get_config.return_value = mock_config

            captioner = create_image_captioner()

            assert isinstance(captioner, ImageCaptioner)
            assert captioner.config == mock_config

    def test_create_image_captioner_with_config(self, mock_config):
        """Test creating ImageCaptioner with config."""
        captioner = create_image_captioner(config=mock_config)

        assert isinstance(captioner, ImageCaptioner)
        assert captioner.config == mock_config


class TestImageCaptionerIntegration:
    """Integration tests for image captioning workflow."""

    def test_array_format_handling(self, image_captioner):
        """Test handling different array formats."""
        # Test RGB format (H, W, C)
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Test CHW format (C, H, W)
        chw_array = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)

        # Test float format
        float_array = np.random.rand(100, 100, 3).astype(np.float32)

        with patch.object(image_captioner, "_load_model") as mock_load_model:
            mock_processor = MagicMock()
            mock_model = MagicMock()
            mock_load_model.return_value = (mock_model, mock_processor)

            mock_inputs = {"pixel_values": torch.randn(1, 3, 224, 224)}
            mock_processor.return_value = mock_inputs
            mock_outputs = torch.tensor([[1, 2, 3]])
            mock_model.generate.return_value = mock_outputs
            mock_processor.batch_decode.return_value = ["Test caption"]

            # All formats should work without raising errors
            result1 = image_captioner.caption_image(image_array=rgb_array)
            result2 = image_captioner.caption_image(image_array=chw_array)
            result3 = image_captioner.caption_image(image_array=float_array)

            assert all(
                isinstance(r, CaptionResult) for r in [result1, result2, result3]
            )

    def test_unsupported_array_format(self, image_captioner):
        """Test handling unsupported array formats."""
        # Create unsupported format (grayscale)
        grayscale_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        with pytest.raises(ValueError) as exc_info:
            image_captioner.caption_image(image_array=grayscale_array)

        assert "Unsupported image array shape" in str(exc_info.value)
