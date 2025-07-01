"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from deep_brief.utils.config import (
    DeepBriefConfig,
    load_config,
    save_config,
    setup_logging,
)


def test_default_config() -> None:
    """Test that default configuration loads correctly."""
    config = DeepBriefConfig()

    assert config.app_name == "DeepBrief"
    assert config.debug is False
    assert config.processing.max_video_size_mb == 500
    assert config.scene_detection.threshold == 0.4
    assert config.transcription.model == "whisper-base"
    assert config.logging.level == "INFO"


def test_config_validation() -> None:
    """Test configuration validation."""
    # Test invalid video format
    with pytest.raises(ValueError, match="Unsupported format"):
        DeepBriefConfig(processing={"supported_formats": ["invalid"]})

    # Test invalid WPM range
    with pytest.raises(ValueError, match="WPM min must be less than max"):
        DeepBriefConfig(analysis={"target_wpm_range": [160, 140]})

    # Test invalid log level
    with pytest.raises(ValueError, match="Invalid log level"):
        DeepBriefConfig(logging={"level": "INVALID"})


def test_load_config_from_file(temp_dir: Path) -> None:
    """Test loading configuration from YAML file."""
    config_file = temp_dir / "test_config.yaml"
    config_data = {
        "app_name": "Test App",
        "debug": True,
        "processing": {"max_video_size_mb": 100},
        "logging": {"level": "DEBUG"},
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    assert config.app_name == "Test App"
    assert config.debug is True
    assert config.processing.max_video_size_mb == 100
    assert config.logging.level == "DEBUG"


def test_save_config(temp_dir: Path) -> None:
    """Test saving configuration to file."""
    config = DeepBriefConfig(
        app_name="Test Save", debug=True, processing={"max_video_size_mb": 250}
    )

    config_file = temp_dir / "saved_config.yaml"
    save_config(config, config_file)

    assert config_file.exists()

    # Load and verify
    loaded_config = load_config(config_file)
    assert loaded_config.app_name == "Test Save"
    assert loaded_config.debug is True
    assert loaded_config.processing.max_video_size_mb == 250


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides."""
    monkeypatch.setenv("DEEP_BRIEF_APP_NAME", "Env Override")
    monkeypatch.setenv("DEEP_BRIEF_DEBUG", "true")
    monkeypatch.setenv("DEEP_BRIEF_PROCESSING__MAX_VIDEO_SIZE_MB", "750")

    config = DeepBriefConfig()

    assert config.app_name == "Env Override"
    assert config.debug is True
    assert config.processing.max_video_size_mb == 750


def test_nested_config_validation() -> None:
    """Test validation of nested configuration objects."""
    config = DeepBriefConfig()

    # Test scene detection bounds
    assert 0.1 <= config.scene_detection.threshold <= 0.9
    assert config.scene_detection.min_scene_duration >= 0.5

    # Test audio settings
    assert 8000 <= config.audio.sample_rate <= 48000
    assert 1 <= config.audio.channels <= 2

    # Test analysis settings
    assert len(config.analysis.target_wpm_range) == 2
    assert config.analysis.target_wpm_range[0] < config.analysis.target_wpm_range[1]


def test_setup_logging(temp_dir: Path) -> None:
    """Test logging setup."""
    import logging

    log_file = temp_dir / "test.log"

    from deep_brief.utils.config import LoggingConfig

    logging_config = LoggingConfig(
        level="DEBUG",
        file_path=log_file,
        console_enabled=False,  # Don't spam console during tests
    )

    setup_logging(logging_config)

    logger = logging.getLogger("deep_brief")
    logger.debug("Test debug message")
    logger.info("Test info message")

    assert log_file.exists()
    log_content = log_file.read_text()
    assert "Test debug message" in log_content
    assert "Test info message" in log_content


def test_config_file_precedence(temp_dir: Path) -> None:
    """Test configuration file precedence."""
    # Create config in temp dir
    config_file = temp_dir / "config.yaml"
    config_data = {"app_name": "File Config", "debug": False}

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.app_name == "File Config"
    assert config.debug is False


def test_missing_config_file() -> None:
    """Test handling of missing config file."""
    nonexistent_file = Path("nonexistent_config.yaml")
    config = load_config(nonexistent_file)

    # Should still work with defaults
    assert config.app_name == "DeepBrief"
    assert config.debug is False


def test_malformed_config_file(temp_dir: Path) -> None:
    """Test handling of malformed config file."""
    config_file = temp_dir / "malformed.yaml"
    config_file.write_text("invalid: yaml: content: [")

    # Should not crash, should use defaults
    config = load_config(config_file)
    assert config.app_name == "DeepBrief"


@pytest.mark.parametrize(
    "model_name",
    [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        "whisper-large",
    ],
)
def test_valid_whisper_models(model_name: str) -> None:
    """Test that all valid Whisper models are accepted."""
    config = DeepBriefConfig(transcription={"model": model_name})
    assert config.transcription.model == model_name


def test_invalid_whisper_model() -> None:
    """Test that invalid Whisper models are rejected."""
    with pytest.raises(ValueError, match="Invalid model"):
        DeepBriefConfig(transcription={"model": "invalid-model"})
