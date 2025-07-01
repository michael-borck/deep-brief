"""Configuration management for DeepBrief."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    max_video_size_mb: int = Field(default=500, ge=1, le=5000)
    supported_formats: list[str] = Field(
        default=["mp4", "mov", "avi", "webm"], min_length=1
    )
    temp_dir: Path = Field(default=Path("temp"))
    cleanup_temp_files: bool = Field(default=True)

    @field_validator("supported_formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate supported video formats."""
        valid_formats = {"mp4", "mov", "avi", "webm", "mkv", "flv"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Unsupported format: {fmt}")
        return [fmt.lower() for fmt in v]


class SceneDetectionConfig(BaseModel):
    """Scene detection configuration."""

    method: str = Field(default="threshold", pattern="^(threshold|adaptive)$")
    threshold: float = Field(default=0.4, ge=0.1, le=0.9)
    min_scene_duration: float = Field(default=2.0, ge=0.5, le=30.0)
    fallback_interval: float = Field(default=30.0, ge=5.0, le=300.0)


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    noise_reduction: bool = Field(default=False)
    normalize_audio: bool = Field(default=True)


class TranscriptionConfig(BaseModel):
    """Speech transcription configuration."""

    model: str = Field(default="whisper-base")
    language: str = Field(default="auto")
    word_timestamps: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    device: str = Field(default="auto")  # auto, cpu, cuda

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name."""
        valid_models = {
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large",
            "whisper-large-v2",
            "whisper-large-v3",
        }
        if v not in valid_models:
            raise ValueError(f"Invalid model: {v}. Valid options: {valid_models}")
        return v


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    filler_words: list[str] = Field(
        default=["um", "uh", "like", "you know", "so", "actually", "basically"]
    )
    target_wpm_range: list[int] = Field(default=[140, 160], min_length=2, max_length=2)
    sentiment_analysis: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("target_wpm_range")
    @classmethod
    def validate_wpm_range(cls, v: list[int]) -> list[int]:
        """Validate WPM range."""
        if len(v) != 2:
            raise ValueError("WPM range must have exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError("WPM min must be less than max")
        if v[0] < 50 or v[1] > 300:
            raise ValueError("WPM values must be between 50 and 300")
        return v


class VisualAnalysisConfig(BaseModel):
    """Visual analysis configuration."""

    # Frame extraction settings
    frames_per_scene: int = Field(default=3, ge=1, le=10)
    frame_quality: int = Field(default=85, ge=50, le=100)
    frame_format: str = Field(default="JPEG", pattern="^(JPEG|PNG|WEBP)$")
    max_frame_width: int = Field(default=1920, ge=480, le=4096)
    max_frame_height: int = Field(default=1080, ge=360, le=2160)

    # Quality assessment thresholds
    blur_threshold: float = Field(default=100.0, ge=10.0, le=1000.0)
    contrast_threshold: float = Field(default=20.0, ge=5.0, le=100.0)
    brightness_min: float = Field(default=50.0, ge=0.0, le=255.0)
    brightness_max: float = Field(default=200.0, ge=0.0, le=255.0)

    # Quality scoring weights
    blur_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    contrast_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    brightness_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Processing settings
    enable_quality_filtering: bool = Field(default=True)
    min_quality_score: float = Field(default=0.3, ge=0.0, le=1.0)
    save_extracted_frames: bool = Field(default=False)

    # Image captioning settings
    enable_captioning: bool = Field(default=True)
    captioning_model: str = Field(default="Salesforce/blip2-opt-2.7b")
    captioning_device: str = Field(default="auto")  # auto, cpu, cuda, mps
    max_caption_length: int = Field(default=50, ge=10, le=200)
    caption_temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    caption_batch_size: int = Field(default=1, ge=1, le=8)

    # OCR settings
    enable_ocr: bool = Field(default=True)
    ocr_engine: str = Field(default="tesseract")  # tesseract, easyocr
    ocr_languages: list[str] = Field(default=["eng"])
    ocr_confidence_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    ocr_text_min_length: int = Field(default=3, ge=1, le=50)
    detect_slide_text: bool = Field(default=True)
    detect_ui_text: bool = Field(default=False)
    
    # Object detection settings
    enable_object_detection: bool = Field(default=True)
    object_detection_model: str = Field(default="heuristic")  # heuristic, yolov5, etc.
    object_detection_device: str = Field(default="auto")  # auto, cpu, cuda, mps
    object_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("blur_weight", "contrast_weight", "brightness_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Ensure quality weights are reasonable."""
        if "blur_weight" in info.data and "contrast_weight" in info.data:
            total = (
                info.data.get("blur_weight", 0)
                + info.data.get("contrast_weight", 0)
                + v
            )
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError("Quality weights must sum to 1.0")
        return v

    @field_validator("captioning_model")
    @classmethod
    def validate_captioning_model(cls, v: str) -> str:
        """Validate image captioning model name."""
        valid_models = {
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-6.7b",
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
        }
        if v not in valid_models:
            # Allow custom models but log a warning
            import logging

            logging.getLogger(__name__).warning(
                f"Using custom captioning model: {v}. Supported models: {valid_models}"
            )
        return v

    @field_validator("ocr_engine")
    @classmethod
    def validate_ocr_engine(cls, v: str) -> str:
        """Validate OCR engine name."""
        valid_engines = {"tesseract", "easyocr"}
        if v.lower() not in valid_engines:
            raise ValueError(f"Invalid OCR engine: {v}. Valid options: {valid_engines}")
        return v.lower()

    @field_validator("ocr_languages")
    @classmethod
    def validate_ocr_languages(cls, v: list[str]) -> list[str]:
        """Validate OCR language codes."""
        # Common language codes for Tesseract
        common_languages = {
            "eng",
            "spa",
            "fra",
            "deu",
            "ita",
            "por",
            "rus",
            "jpn",
            "kor",
            "chi_sim",
            "chi_tra",
        }
        validated = []
        for lang in v:
            if lang.lower() in common_languages:
                validated.append(lang.lower())
            else:
                # Allow custom language codes but log a warning
                import logging

                logging.getLogger(__name__).warning(
                    f"Using custom OCR language: {lang}. Common languages: {common_languages}"
                )
                validated.append(lang.lower())
        return validated


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[str] = Field(default=["json", "html"], min_length=1)
    include_frames: bool = Field(default=True)
    frame_quality: int = Field(default=80, ge=1, le=100)
    report_template: str = Field(default="default")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate output formats."""
        valid_formats = {"json", "html", "pdf", "csv", "markdown"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Unsupported format: {fmt}")
        return [fmt.lower() for fmt in v]


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_enabled: bool = Field(default=True)
    file_path: Path = Field(default=Path("logs/deep_brief.log"))
    max_bytes: int = Field(default=10_000_000)  # 10MB
    backup_count: int = Field(default=5)
    console_enabled: bool = Field(default=True)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class DeepBriefConfig(BaseSettings):
    """Main configuration for DeepBrief application."""

    model_config = SettingsConfigDict(
        env_prefix="DEEP_BRIEF_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Application settings
    app_name: str = Field(default="DeepBrief")
    version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # Component configurations
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    scene_detection: SceneDetectionConfig = Field(default_factory=SceneDetectionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    visual_analysis: VisualAnalysisConfig = Field(default_factory=VisualAnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Path | None = None) -> DeepBriefConfig:
    """
    Load configuration from file and environment variables.

    Args:
        config_path: Path to YAML config file. If None, uses default locations.

    Returns:
        Loaded configuration object.
    """
    config_data: dict[str, Any] = {}

    # Try to load from YAML file
    if config_path is None:
        # Check default locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path.home() / ".deep-brief" / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")

    # Create config object (environment variables will override file values)
    return DeepBriefConfig(**config_data)


def save_config(config: DeepBriefConfig, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        config_path: Path where to save the config file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding computed fields and converting Path objects to strings
    config_dict = config.model_dump(exclude={"version"}, mode="json")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging based on configuration.

    Args:
        config: Logging configuration.
    """
    import logging.handlers

    # Create logger
    logger = logging.getLogger("deep_brief")
    logger.setLevel(getattr(logging, config.level))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if config.file_enabled:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set library loggers to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


# Global configuration cache
_global_config: DeepBriefConfig | None = None


def get_config() -> DeepBriefConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
        setup_logging(_global_config.logging)
    return _global_config


def reload_config(_config_path: Path | None = None) -> DeepBriefConfig:
    """Reload configuration from file."""
    if hasattr(get_config, "_config"):
        delattr(get_config, "_config")
    return get_config()
