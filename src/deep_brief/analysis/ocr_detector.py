"""OCR functionality for text detection in video frames.

This module provides OCR capabilities for detecting and extracting text
from presentation slides and other visual content in video frames.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from deep_brief.analysis.error_handling import (
    ImageValidationError,
    ModelInitializationError,
    validate_image,
    with_retry,
)
from deep_brief.core.exceptions import ErrorCode, VideoProcessingError
from deep_brief.utils.config import get_config

logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning, module="easyocr")


class TextRegion(BaseModel):
    """Detected text region with position and content."""

    text: str
    confidence: float  # OCR confidence (0-100)
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    language: str | None = None
    font_size_estimate: float | None = None
    is_title: bool = False  # Heuristic for title/header text
    is_slide_number: bool = False  # Heuristic for slide numbers


class OCRResult(BaseModel):
    """Result of OCR operation on an image."""

    text_regions: list[TextRegion]
    full_text: str  # Concatenated text from all regions
    processing_time: float  # Time taken for OCR (seconds)
    engine_used: str
    languages_detected: list[str]
    total_text_regions: int
    high_confidence_regions: int  # Regions above confidence threshold
    average_confidence: float


class OCRDetector:
    """OCR text detection using Tesseract or EasyOCR."""

    def __init__(self, config: Any = None):
        """Initialize OCR detector with configuration."""
        self.config = config or get_config()
        self.engine = self.config.visual_analysis.ocr_engine
        self.languages = self.config.visual_analysis.ocr_languages
        self.confidence_threshold = self.config.visual_analysis.ocr_confidence_threshold
        self.min_text_length = self.config.visual_analysis.ocr_text_min_length

        # Initialize engine-specific components
        self.tesseract_config = None
        self.easyocr_reader = None

        self._validate_dependencies()
        self._initialize_engine()

        logger.info(f"OCRDetector initialized with engine: {self.engine}")

    def _validate_dependencies(self):
        """Validate that required OCR dependencies are available."""
        if self.engine == "tesseract" and not TESSERACT_AVAILABLE:
            raise VideoProcessingError(
                message="Tesseract OCR not available. Install with: pip install pytesseract",
                error_code=ErrorCode.MISSING_DEPENDENCY,
                details={"missing_package": "pytesseract"},
            )

        if self.engine == "easyocr" and not EASYOCR_AVAILABLE:
            raise VideoProcessingError(
                message="EasyOCR not available. Install with: pip install easyocr",
                error_code=ErrorCode.MISSING_DEPENDENCY,
                details={"missing_package": "easyocr"},
            )

    def _initialize_engine(self):
        """Initialize the specific OCR engine."""
        if self.engine == "tesseract":
            self._initialize_tesseract()
        elif self.engine == "easyocr":
            self._initialize_easyocr()

    @with_retry(max_attempts=2, delay=1.0)
    def _initialize_tesseract(self):
        """Initialize Tesseract OCR configuration."""
        try:
            # Test Tesseract availability
            pytesseract.get_tesseract_version()

            # Configure Tesseract
            lang_string = "+".join(self.languages)
            self.tesseract_config = f"-l {lang_string} --oem 3 --psm 6"

            logger.info(f"Tesseract initialized with languages: {self.languages}")

        except Exception as e:
            raise ModelInitializationError(
                message=f"Failed to initialize Tesseract: {str(e)}",
                model_name="tesseract",
                details={"languages": self.languages},
                cause=e,
            ) from e

    @with_retry(max_attempts=2, delay=2.0)
    def _initialize_easyocr(self):
        """Initialize EasyOCR reader."""
        try:
            # Map common language codes to EasyOCR format
            easyocr_languages = []
            lang_mapping = {
                "eng": "en",
                "spa": "es",
                "fra": "fr",
                "deu": "de",
                "ita": "it",
                "por": "pt",
                "rus": "ru",
                "jpn": "ja",
                "kor": "ko",
                "chi_sim": "ch_sim",
                "chi_tra": "ch_tra",
            }

            for lang in self.languages:
                mapped_lang = lang_mapping.get(lang, lang)
                easyocr_languages.append(mapped_lang)

            self.easyocr_reader = easyocr.Reader(easyocr_languages, gpu=False)
            logger.info(f"EasyOCR initialized with languages: {easyocr_languages}")

        except Exception as e:
            raise ModelInitializationError(
                message=f"Failed to initialize EasyOCR: {str(e)}",
                model_name="easyocr",
                details={"languages": easyocr_languages},
                cause=e,
            ) from e

    def detect_text(
        self,
        image_path: Path | None = None,
        image_array: np.ndarray | None = None,
        pil_image: Image.Image | None = None,
        preprocess: bool = True,
    ) -> OCRResult:
        """
        Detect text in an image using OCR.

        Args:
            image_path: Path to image file
            image_array: Numpy array representation of image
            pil_image: PIL Image object
            preprocess: Whether to apply image preprocessing for better OCR

        Returns:
            OCRResult with detected text and metadata

        Raises:
            VideoProcessingError: If OCR fails
            ValueError: If no image input is provided
        """
        import time

        start_time = time.time()

        # Validate inputs
        if not any([image_path, image_array is not None, pil_image]):
            raise ValueError(
                "Must provide one of: image_path, image_array, or pil_image"
            )

        if not self.config.visual_analysis.enable_ocr:
            # Return empty result if OCR is disabled
            return OCRResult(
                text_regions=[],
                full_text="",
                processing_time=0.0,
                engine_used="none",
                languages_detected=[],
                total_text_regions=0,
                high_confidence_regions=0,
                average_confidence=0.0,
            )

        logger.debug(f"Detecting text with engine: {self.engine}")

        try:
            # Prepare and validate image
            if pil_image is not None:
                image_array_tmp = np.array(pil_image)
                validated_array = validate_image(image_array_tmp, "PIL image")
                image = Image.fromarray(validated_array)
            elif image_path is not None:
                if not image_path.exists():
                    raise VideoProcessingError(
                        message=f"Image file not found: {image_path}",
                        error_code=ErrorCode.FILE_NOT_FOUND,
                        file_path=image_path,
                    )
                validated_array = validate_image(image_path, f"image file {image_path}")
                image = Image.fromarray(validated_array)
            else:
                # Validate numpy array
                validated_array = validate_image(image_array, "numpy array")
                image = Image.fromarray(validated_array)

            # Preprocess image for better OCR if requested
            if preprocess:
                image = self._preprocess_image(image)

            # Perform OCR based on engine
            if self.engine == "tesseract":
                text_regions = self._detect_text_tesseract(image)
            elif self.engine == "easyocr":
                text_regions = self._detect_text_easyocr(image)
            else:
                raise ValueError(f"Unsupported OCR engine: {self.engine}")

            # Filter by confidence and text length
            filtered_regions = self._filter_text_regions(text_regions)

            # Analyze text regions for semantic information
            analyzed_regions = self._analyze_text_regions(filtered_regions)

            # Create full text by concatenating regions
            full_text = " ".join([region.text for region in analyzed_regions])

            # Calculate statistics
            total_regions = len(analyzed_regions)
            high_confidence_regions = sum(
                1
                for region in analyzed_regions
                if region.confidence >= self.confidence_threshold
            )
            average_confidence = (
                sum(region.confidence for region in analyzed_regions) / total_regions
                if total_regions > 0
                else 0.0
            )

            # Detect languages present
            languages_detected = list(
                set(region.language for region in analyzed_regions if region.language)
            )

            processing_time = time.time() - start_time

            result = OCRResult(
                text_regions=analyzed_regions,
                full_text=full_text.strip(),
                processing_time=processing_time,
                engine_used=self.engine,
                languages_detected=languages_detected,
                total_text_regions=total_regions,
                high_confidence_regions=high_confidence_regions,
                average_confidence=average_confidence,
            )

            logger.debug(
                f"OCR completed: {total_regions} regions, {len(full_text)} chars "
                f"(avg confidence: {average_confidence:.1f}%, processing time: {processing_time:.1f}s)"
            )

            return result

        except VideoProcessingError:
            # Re-raise VideoProcessingError as-is (preserves original error code)
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR detection failed after {processing_time:.1f}s: {str(e)}"
            logger.error(error_msg)

            raise VideoProcessingError(
                message=error_msg,
                error_code=ErrorCode.FRAME_EXTRACTION_FAILED,  # Generic visual processing error
                details={"engine": self.engine, "processing_time": processing_time},
                cause=e,
            ) from e

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to OpenCV format for preprocessing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to handle varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Noise removal with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.medianBlur(processed, 3)

        # Convert back to PIL Image
        return Image.fromarray(processed)

    def _detect_text_tesseract(self, image: Image.Image) -> list[TextRegion]:
        """Detect text using Tesseract OCR."""
        text_regions = []

        try:
            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(
                image, config=self.tesseract_config, output_type=pytesseract.Output.DICT
            )

            n_boxes = len(data["level"])
            for i in range(n_boxes):
                # Skip if confidence is too low or text is empty
                confidence = float(data["conf"][i])
                text = data["text"][i].strip()

                if confidence < 0 or not text:
                    continue

                # Extract bounding box
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                text_region = TextRegion(
                    text=text,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    language=self.languages[0] if self.languages else None,
                )

                text_regions.append(text_region)

        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")

        return text_regions

    def _detect_text_easyocr(self, image: Image.Image) -> list[TextRegion]:
        """Detect text using EasyOCR."""
        text_regions = []

        try:
            # Convert PIL to numpy array for EasyOCR
            image_array = np.array(image)

            # Perform OCR
            results = self.easyocr_reader.readtext(image_array)

            for result in results:
                bbox_points, text, confidence = result

                # Convert confidence to percentage
                confidence_pct = confidence * 100

                # Extract bounding box coordinates
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - min(x_coords))
                h = int(max(y_coords) - min(y_coords))

                text_region = TextRegion(
                    text=text.strip(),
                    confidence=confidence_pct,
                    bbox=(x, y, w, h),
                    language=None,  # EasyOCR doesn't provide language per region
                )

                text_regions.append(text_region)

        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")

        return text_regions

    def _filter_text_regions(self, text_regions: list[TextRegion]) -> list[TextRegion]:
        """Filter text regions by confidence and length."""
        filtered = []

        for region in text_regions:
            # Filter by confidence threshold
            if region.confidence < self.confidence_threshold:
                continue

            # Filter by minimum text length
            if len(region.text) < self.min_text_length:
                continue

            # Skip pure whitespace or non-printable characters
            if not region.text.strip() or not any(c.isprintable() for c in region.text):
                continue

            filtered.append(region)

        return filtered

    def _analyze_text_regions(self, text_regions: list[TextRegion]) -> list[TextRegion]:
        """Analyze text regions for semantic information."""
        if not text_regions:
            return text_regions

        # Calculate font size estimates based on bounding box height
        heights = [region.bbox[3] for region in text_regions]
        avg_height = sum(heights) / len(heights) if heights else 0

        for region in text_regions:
            # Estimate font size relative to average
            bbox_height = region.bbox[3]
            region.font_size_estimate = (
                bbox_height / avg_height if avg_height > 0 else 1.0
            )

            # Heuristic for title detection (larger font, position at top)
            if bbox_height > avg_height * 1.5 and region.bbox[1] < avg_height * 2:
                region.is_title = True

            # Heuristic for slide number detection (small text, numbers, bottom/corner)
            text_lower = region.text.lower().strip()
            if (
                any(char.isdigit() for char in text_lower)
                and len(text_lower) <= 10
                and bbox_height < avg_height * 0.8
            ):
                # Check if positioned in typical slide number locations
                image_height = max(r.bbox[1] + r.bbox[3] for r in text_regions)
                if region.bbox[1] > image_height * 0.8:  # Bottom area
                    region.is_slide_number = True

        return text_regions

    def detect_text_batch(
        self, images: list[Path | Image.Image], preprocess: bool = True, **kwargs
    ) -> list[OCRResult]:
        """
        Detect text in a batch of images.

        Args:
            images: List of image paths or PIL Images
            preprocess: Whether to apply preprocessing
            **kwargs: Additional arguments passed to detect_text

        Returns:
            List of OCRResult objects
        """
        results = []

        for image in images:
            try:
                if isinstance(image, Path):
                    result = self.detect_text(
                        image_path=image, preprocess=preprocess, **kwargs
                    )
                else:
                    result = self.detect_text(
                        pil_image=image, preprocess=preprocess, **kwargs
                    )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to detect text in image {image}: {e}")
                # Add failed result
                results.append(
                    OCRResult(
                        text_regions=[],
                        full_text=f"OCR failed: {str(e)}",
                        processing_time=0.0,
                        engine_used=self.engine,
                        languages_detected=[],
                        total_text_regions=0,
                        high_confidence_regions=0,
                        average_confidence=0.0,
                    )
                )

        return results

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes for the current engine."""
        if self.engine == "tesseract":
            try:
                # Get available languages from Tesseract
                langs = pytesseract.get_languages(config="")
                return langs
            except Exception:
                # Return common language codes if detection fails
                return ["eng", "spa", "fra", "deu", "ita", "por", "rus", "jpn", "kor"]

        elif self.engine == "easyocr":
            # EasyOCR supported languages
            return [
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "ja",
                "ko",
                "ch_sim",
                "ch_tra",
                "ar",
                "hi",
                "th",
                "vi",
            ]

        return []

    def cleanup(self):
        """Clean up OCR resources."""
        if self.easyocr_reader is not None:
            # EasyOCR doesn't have explicit cleanup, but we can clear the reference
            self.easyocr_reader = None
            logger.info("EasyOCR resources cleaned up")


def create_ocr_detector(config: Any = None) -> OCRDetector:
    """Create a new OCRDetector instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured OCRDetector instance
    """
    return OCRDetector(config=config)
