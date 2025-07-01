"""Error handling utilities for visual analysis components.

This module provides robust error handling for image processing,
model loading, and analysis operations.
"""

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np
from PIL import Image

from deep_brief.core.exceptions import ErrorCode, VideoProcessingError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ImageValidationError(VideoProcessingError):
    """Error raised when image validation fails."""
    
    def __init__(
        self,
        message: str,
        image_path: Path | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
            file_path=image_path,
            details=details or {},
            cause=cause,
        )


class ModelInitializationError(VideoProcessingError):
    """Error raised when model initialization fails."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOADING_FAILED,
            details={"model_name": model_name, **(details or {})},
            cause=cause,
        )


def validate_image(image: Any, source: str = "image") -> np.ndarray:
    """
    Validate and normalize image input.
    
    Args:
        image: Image input (numpy array, PIL Image, or path)
        source: Description of image source for error messages
        
    Returns:
        Validated numpy array in RGB format
        
    Raises:
        ImageValidationError: If image is invalid or corrupted
    """
    try:
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise ImageValidationError(
                    f"Image file not found: {image_path}",
                    image_path=image_path,
                )
            
            # Try to load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ImageValidationError(
                    f"Failed to load image file: {image_path}",
                    image_path=image_path,
                    cause=e,
                )
        
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Validate numpy array
        if not isinstance(image, np.ndarray):
            raise ImageValidationError(
                f"Invalid image type: {type(image).__name__}. Expected numpy array or PIL Image."
            )
        
        # Check if image is empty first
        if image.size == 0 or 0 in image.shape:
            raise ImageValidationError(
                "Image is empty or has zero dimensions",
                details={"shape": image.shape},
            )
        
        # Check array shape
        if image.ndim < 2 or image.ndim > 4:
            raise ImageValidationError(
                f"Invalid image dimensions: {image.ndim}. Expected 2D, 3D, or 4D array.",
                details={"shape": image.shape},
            )
        
        
        # Check data type and range
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            logger.warning(f"Unusual image dtype: {image.dtype}. Converting to uint8.")
            
        # Normalize to uint8 if needed
        if image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                # Assume float images are in [0, 1] range
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                # Scale other types to uint8
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert to RGB if needed
        if image.ndim == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                # Single channel to RGB
                image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
            elif image.shape[2] != 3:
                raise ImageValidationError(
                    f"Invalid number of channels: {image.shape[2]}. Expected 1, 3, or 4.",
                    details={"shape": image.shape},
                )
        elif image.ndim == 4:
            # Batch dimension - take first image
            logger.warning("Batch dimension detected, using first image only")
            image = image[0]
            return validate_image(image, source)  # Recursive validation
        
        # Final validation
        if image.shape[0] < 10 or image.shape[1] < 10:
            raise ImageValidationError(
                f"Image too small: {image.shape[:2]}. Minimum size is 10x10.",
                details={"shape": image.shape},
            )
        
        return image
        
    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(
            f"Unexpected error validating {source}",
            details={"error_type": type(e).__name__},
            cause=e,
        )


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff: Multiplier for delay after each failure
        exceptions: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            # Re-raise the last exception
            if last_exception:
                raise last_exception
            
        return wrapper
    return decorator


def safe_model_inference(
    func: Callable[..., T],
    fallback_result: T,
    operation_name: str = "model inference",
) -> Callable[..., T]:
    """
    Decorator for safe model inference with fallback.
    
    Args:
        func: Function to wrap
        fallback_result: Result to return if inference fails
        operation_name: Name of operation for logging
        
    Returns:
        Wrapped function that returns fallback on failure
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{operation_name} failed: {e}. Returning fallback result.")
            return fallback_result
    
    return wrapper


def validate_model_inputs(**validators: Callable[[Any], Any]) -> Callable:
    """
    Decorator to validate model inputs before inference.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validation functions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate specified arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    try:
                        bound_args.arguments[param_name] = validator(
                            bound_args.arguments[param_name]
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}': {e}"
                        ) from e
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator


class ErrorRecoveryContext:
    """Context manager for error recovery in processing pipelines."""
    
    def __init__(
        self,
        operation_name: str,
        fallback_action: Callable[[], Any] | None = None,
        suppress_errors: bool = False,
    ):
        """
        Initialize error recovery context.
        
        Args:
            operation_name: Name of the operation for logging
            fallback_action: Optional function to call on error
            suppress_errors: Whether to suppress exceptions
        """
        self.operation_name = operation_name
        self.fallback_action = fallback_action
        self.suppress_errors = suppress_errors
        self.start_time = None
        self.error = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context with error handling."""
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            logger.info(f"Completed {self.operation_name} in {elapsed:.2f}s")
            return False
        
        self.error = exc_val
        logger.error(
            f"{self.operation_name} failed after {elapsed:.2f}s: {exc_val}",
            exc_info=True,
        )
        
        # Execute fallback action if provided
        if self.fallback_action:
            try:
                logger.info(f"Executing fallback for {self.operation_name}")
                self.fallback_action()
            except Exception as fallback_error:
                logger.error(f"Fallback action failed: {fallback_error}")
        
        # Suppress error if requested
        return self.suppress_errors


def handle_corrupt_frame(
    frame: np.ndarray,
    frame_info: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """
    Handle potentially corrupt frame data.
    
    Args:
        frame: Frame data to check
        frame_info: Optional frame metadata
        
    Returns:
        Repaired frame or None if unrecoverable
    """
    try:
        # Validate frame
        validated_frame = validate_image(frame, "frame")
        
        # Check for common corruption patterns
        if np.all(validated_frame == 0):
            logger.warning("Frame is completely black")
            return None
        
        if np.all(validated_frame == 255):
            logger.warning("Frame is completely white")
            return None
        
        # Check for excessive noise
        gray = cv2.cvtColor(validated_frame, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 10000:  # Very high variance indicates noise
            logger.warning(f"Frame appears to be corrupted (noise variance: {laplacian_var:.1f})")
            # Try denoising
            try:
                denoised = cv2.fastNlMeansDenoisingColored(validated_frame, None, 10, 10, 7, 21)
                logger.info("Applied denoising to corrupted frame")
                return denoised
            except Exception as e:
                logger.error(f"Denoising failed: {e}")
                return None
            else:
                logger.info("Applied denoising to corrupted frame")
        
        return validated_frame
        
    except Exception as e:
        logger.error(f"Frame validation failed: {e}")
        if frame_info:
            logger.error(f"Frame info: {frame_info}")
        return None