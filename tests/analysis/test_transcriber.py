"""Tests for speech transcription functionality."""

from unittest.mock import MagicMock, patch

import pytest

from deep_brief.analysis.transcriber import (
    LanguageDetectionResult,
    Segment,
    TranscriptionResult,
    WhisperTranscriber,
    WordTimestamp,
    create_transcriber,
)
from deep_brief.core.audio_extractor import AudioInfo
from deep_brief.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
    VideoProcessingError,
)
from deep_brief.utils.config import DeepBriefConfig, TranscriptionConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = DeepBriefConfig(
        transcription=TranscriptionConfig(
            model="whisper-base",
            language="auto",
            word_timestamps=True,
            temperature=0.0,
            device="cpu",
        )
    )
    return config


@pytest.fixture
def transcriber(mock_config):
    """Create WhisperTranscriber instance for testing."""
    return WhisperTranscriber(config=mock_config)


@pytest.fixture
def mock_audio_info(tmp_path):
    """Create mock AudioInfo for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_text("fake audio content")

    return AudioInfo(
        file_path=audio_file,
        duration=30.0,
        sample_rate=16000,
        channels=1,
        size_mb=5.0,
        format="wav",
    )


@pytest.fixture
def mock_whisper_result():
    """Mock Whisper transcription result."""
    return {
        "text": "Hello world. This is a test.",
        "language": "en",
        "language_probability": 0.95,
        "segments": [
            {
                "id": 0,
                "text": "Hello world.",
                "start": 0.0,
                "end": 2.0,
                "avg_logprob": -0.5,
                "no_speech_prob": 0.1,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.95},
                    {"word": "world", "start": 0.6, "end": 1.2, "confidence": 0.92},
                    {"word": ".", "start": 1.2, "end": 1.3, "confidence": 0.99},
                ],
            },
            {
                "id": 1,
                "text": "This is a test.",
                "start": 3.0,
                "end": 5.0,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.05,
                "words": [
                    {"word": "This", "start": 3.0, "end": 3.3, "confidence": 0.98},
                    {"word": "is", "start": 3.4, "end": 3.6, "confidence": 0.96},
                    {"word": "a", "start": 3.7, "end": 3.8, "confidence": 0.99},
                    {"word": "test", "start": 3.9, "end": 4.3, "confidence": 0.94},
                    {"word": ".", "start": 4.3, "end": 4.4, "confidence": 0.99},
                ],
            },
        ],
    }


class TestWordTimestamp:
    """Test WordTimestamp model."""

    def test_word_timestamp_creation(self):
        """Test creating WordTimestamp."""
        word = WordTimestamp(word="hello", start=1.0, end=1.5, confidence=0.95)

        assert word.word == "hello"
        assert word.start == 1.0
        assert word.end == 1.5
        assert word.confidence == 0.95


class TestSegment:
    """Test Segment model."""

    def test_segment_creation(self):
        """Test creating Segment."""
        words = [
            WordTimestamp(word="hello", start=0.0, end=0.5, confidence=0.95),
            WordTimestamp(word="world", start=0.6, end=1.0, confidence=0.92),
        ]

        segment = Segment(
            id=0,
            text="hello world",
            start=0.0,
            end=1.0,
            avg_logprob=-0.5,
            no_speech_prob=0.1,
            words=words,
            language="en",
        )

        assert segment.id == 0
        assert segment.text == "hello world"
        assert len(segment.words) == 2
        assert segment.language == "en"


class TestTranscriptionResult:
    """Test TranscriptionResult model and methods."""

    def test_transcription_result_creation(self):
        """Test creating TranscriptionResult."""
        segments = [
            Segment(
                id=0,
                text="Hello world",
                start=0.0,
                end=2.0,
                avg_logprob=-0.5,
                no_speech_prob=0.1,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                    WordTimestamp(word="world", start=0.6, end=1.2, confidence=0.92),
                ],
            )
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=30.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=5.0,
        )

        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.word_count == 2

    def test_get_words_at_time(self):
        """Test getting words at specific timestamp."""
        segments = [
            Segment(
                id=0,
                text="Hello world",
                start=0.0,
                end=2.0,
                avg_logprob=-0.5,
                no_speech_prob=0.1,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                    WordTimestamp(word="world", start=0.6, end=1.2, confidence=0.92),
                ],
            )
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=30.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=5.0,
        )

        # Find words near timestamp 0.3 (should find "Hello" which ends at 0.5)
        words = result.get_words_at_time(0.3, tolerance=0.1)
        assert len(words) == 1
        assert words[0].word == "Hello"

        # Find words near timestamp 1.0 (should find "world" which overlaps 0.6-1.2)
        words = result.get_words_at_time(1.0, tolerance=0.1)
        assert len(words) == 1
        assert words[0].word == "world"

    def test_get_text_between_times(self):
        """Test getting text between timestamps."""
        segments = [
            Segment(
                id=0,
                text="Hello world",
                start=0.0,
                end=2.0,
                avg_logprob=-0.5,
                no_speech_prob=0.1,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                    WordTimestamp(word="world", start=0.6, end=1.2, confidence=0.92),
                ],
            ),
            Segment(
                id=1,
                text="This is a test",
                start=3.0,
                end=5.0,
                avg_logprob=-0.3,
                no_speech_prob=0.05,
                words=[
                    WordTimestamp(word="This", start=3.0, end=3.3, confidence=0.98),
                    WordTimestamp(word="is", start=3.4, end=3.6, confidence=0.96),
                    WordTimestamp(word="a", start=3.7, end=3.8, confidence=0.99),
                    WordTimestamp(word="test", start=3.9, end=4.3, confidence=0.94),
                ],
            ),
        ]

        result = TranscriptionResult(
            text="Hello world. This is a test.",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=30.0,
            model_used="whisper-base",
            word_count=6,
            processing_time=5.0,
        )

        # Get text from 0.0 to 1.0 (should include "Hello world")
        text = result.get_text_between_times(0.0, 1.0)
        assert "Hello" in text
        assert "world" in text

        # Get text from 3.5 to 4.0 (should include "is a")
        text = result.get_text_between_times(3.5, 4.0)
        assert "is" in text
        assert "a" in text

    def test_get_speaking_rate(self):
        """Test calculating speaking rate."""
        segments = [
            Segment(
                id=0,
                text="Hello world",
                start=0.0,
                end=60.0,  # 1 minute
                avg_logprob=-0.5,
                no_speech_prob=0.1,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                    WordTimestamp(word="world", start=0.6, end=1.2, confidence=0.92),
                ],
            )
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=60.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=5.0,
        )

        # Speaking rate for full duration (2 words in 1 minute = 2 WPM)
        wpm = result.get_speaking_rate(0.0, 60.0)
        assert wpm == 2.0

        # Speaking rate for 30 seconds (2 words in 0.5 minutes = 4 WPM)
        wpm = result.get_speaking_rate(0.0, 30.0)
        assert wpm == 4.0

    def test_to_dict(self):
        """Test converting result to dictionary."""
        segments = [
            Segment(
                id=0,
                text="Hello world",
                start=0.0,
                end=2.0,
                avg_logprob=-0.5,
                no_speech_prob=0.1,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                ],
            )
        ]

        result = TranscriptionResult(
            text="Hello world",
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=30.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=5.0,
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Hello world"
        assert result_dict["language"] == "en"
        assert result_dict["word_count"] == 2
        assert len(result_dict["segments"]) == 1


class TestWhisperTranscriber:
    """Test WhisperTranscriber class."""

    def test_initialization_default_config(self):
        """Test transcriber initialization with default config."""
        transcriber = WhisperTranscriber()
        assert transcriber.config is not None
        assert transcriber.model is None
        assert transcriber.device in ["cpu", "cuda", "mps"]

    def test_initialization_custom_config(self, mock_config):
        """Test transcriber initialization with custom config."""
        transcriber = WhisperTranscriber(config=mock_config)
        assert transcriber.config == mock_config
        assert transcriber.device == "cpu"  # From mock config

    def test_determine_device_auto_cpu(self, mock_config):
        """Test device determination when CUDA not available."""
        mock_config.transcription.device = "auto"

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            transcriber = WhisperTranscriber(config=mock_config)
            assert transcriber.device == "cpu"

    def test_determine_device_auto_cuda(self, mock_config):
        """Test device determination when CUDA available."""
        mock_config.transcription.device = "auto"

        with patch("torch.cuda.is_available", return_value=True):
            transcriber = WhisperTranscriber(config=mock_config)
            assert transcriber.device == "cuda"

    def test_determine_device_explicit(self, mock_config):
        """Test explicit device configuration."""
        mock_config.transcription.device = "cuda"
        transcriber = WhisperTranscriber(config=mock_config)
        assert transcriber.device == "cuda"

    @patch("whisper.load_model")
    def test_load_model_success(self, mock_load_model, transcriber):
        """Test successful model loading."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        model = transcriber._load_model()

        assert model == mock_model
        assert transcriber.model == mock_model
        mock_load_model.assert_called_once_with("base", device="cpu")

    @patch("whisper.load_model")
    def test_load_model_failure(self, mock_load_model, transcriber):
        """Test model loading failure."""
        mock_load_model.side_effect = RuntimeError("Model not found")

        with pytest.raises(VideoProcessingError) as exc_info:
            transcriber._load_model()

        assert exc_info.value.error_code.value == "MISSING_DEPENDENCY"

    @patch("whisper.load_model")
    def test_transcribe_audio_success(
        self, mock_load_model, transcriber, mock_audio_info, mock_whisper_result
    ):
        """Test successful audio transcription."""
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(mock_audio_info)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world. This is a test."
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.word_count > 0

        # Verify model was called correctly
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        assert str(mock_audio_info.file_path) in call_args[0]

    @patch("whisper.load_model")
    def test_transcribe_audio_with_custom_parameters(
        self, mock_load_model, transcriber, mock_audio_info, mock_whisper_result
    ):
        """Test transcription with custom parameters."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(
            mock_audio_info, language="es", temperature=0.5, word_timestamps=False
        )

        assert isinstance(result, TranscriptionResult)

        # Verify parameters were passed
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "es"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["word_timestamps"] is False

    def test_transcribe_audio_file_not_found(self, transcriber, tmp_path):
        """Test transcription with non-existent audio file."""
        audio_info = AudioInfo(
            file_path=tmp_path / "nonexistent.wav",
            duration=30.0,
            sample_rate=16000,
            channels=1,
            size_mb=5.0,
            format="wav",
        )

        with pytest.raises(AudioProcessingError) as exc_info:
            transcriber.transcribe_audio(audio_info)

        assert exc_info.value.error_code.value == "FILE_NOT_FOUND"

    @patch("whisper.load_model")
    def test_transcribe_audio_whisper_error(
        self, mock_load_model, transcriber, mock_audio_info
    ):
        """Test handling of Whisper transcription errors."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Transcription failed")
        mock_load_model.return_value = mock_model

        with pytest.raises(AudioProcessingError) as exc_info:
            transcriber.transcribe_audio(mock_audio_info)

        assert "Transcription failed" in str(exc_info.value)

    @patch("whisper.load_model")
    def test_transcribe_audio_cuda_memory_error(
        self, mock_load_model, transcriber, mock_audio_info
    ):
        """Test handling of CUDA memory errors."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")
        mock_load_model.return_value = mock_model

        with pytest.raises(AudioProcessingError) as exc_info:
            transcriber.transcribe_audio(mock_audio_info)

        assert exc_info.value.error_code.value == "INSUFFICIENT_MEMORY"

    @patch("whisper.load_model")
    def test_transcribe_audio_segment(
        self, mock_load_model, transcriber, mock_audio_info, mock_whisper_result
    ):
        """Test transcribing audio segment."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio_segment(
            mock_audio_info, start_time=1.0, end_time=4.0
        )

        assert isinstance(result, TranscriptionResult)
        assert result.duration == 3.0  # end_time - start_time

        # Should filter segments to time range
        assert len(result.segments) > 0

    def test_get_supported_languages(self, transcriber):
        """Test getting supported languages."""
        languages = transcriber.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages

    @patch("torch.cuda.empty_cache")
    def test_cleanup_cuda(self, mock_empty_cache, mock_config):
        """Test cleanup with CUDA device."""
        mock_config.transcription.device = "cuda"
        transcriber = WhisperTranscriber(config=mock_config)
        transcriber.model = MagicMock()  # Set a mock model

        with patch("torch.cuda.is_available", return_value=True):
            transcriber.cleanup()

        assert transcriber.model is None
        mock_empty_cache.assert_called_once()

    def test_cleanup_cpu(self, transcriber):
        """Test cleanup with CPU device."""
        transcriber.model = MagicMock()  # Set a mock model

        transcriber.cleanup()

        assert transcriber.model is None


class TestTranscriberFactory:
    """Test transcriber factory function."""

    def test_create_transcriber_no_config(self):
        """Test creating transcriber without config."""
        transcriber = create_transcriber()

        assert isinstance(transcriber, WhisperTranscriber)
        assert transcriber.config is not None

    def test_create_transcriber_with_config(self, mock_config):
        """Test creating transcriber with config."""
        transcriber = create_transcriber(config=mock_config)

        assert isinstance(transcriber, WhisperTranscriber)
        assert transcriber.config == mock_config


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_language_detection_result_creation(self):
        """Test creating LanguageDetectionResult."""
        result = LanguageDetectionResult(
            detected_language="en",
            confidence=0.95,
            all_probabilities={"en": 0.95, "es": 0.03, "fr": 0.02},
            detection_method="whisper",
        )

        assert result.detected_language == "en"
        assert result.confidence == 0.95
        assert result.detection_method == "whisper"
        assert len(result.all_probabilities) == 3

    @patch("whisper.load_model")
    @patch("whisper.load_audio")
    @patch("whisper.pad_or_trim")
    @patch("whisper.log_mel_spectrogram")
    def test_detect_language_success(
        self,
        mock_mel,
        mock_pad,
        mock_load_audio,
        mock_load_model,
        transcriber,
        mock_audio_info,
    ):
        """Test successful language detection."""
        # Mock Whisper components
        mock_model = MagicMock()
        mock_model.detect_language.return_value = (
            None,
            {"en": 0.95, "es": 0.03, "fr": 0.02},
        )
        mock_model.device = "cpu"
        mock_load_model.return_value = mock_model

        mock_load_audio.return_value = "mock_audio"
        mock_pad.return_value = "padded_audio"
        mock_mel_spec = MagicMock()
        mock_mel_spec.to.return_value = mock_mel_spec
        mock_mel.return_value = mock_mel_spec

        result = transcriber.detect_language(mock_audio_info)

        assert isinstance(result, LanguageDetectionResult)
        assert result.detected_language == "en"
        assert result.confidence == 0.95
        assert result.detection_method == "whisper"
        assert "en" in result.all_probabilities

    def test_detect_language_file_not_found(self, transcriber, tmp_path):
        """Test language detection with non-existent file."""
        audio_info = AudioInfo(
            file_path=tmp_path / "nonexistent.wav",
            duration=30.0,
            sample_rate=16000,
            channels=1,
            size_mb=5.0,
            format="wav",
        )

        with pytest.raises(AudioProcessingError) as exc_info:
            transcriber.detect_language(audio_info)

        assert exc_info.value.error_code.value == "FILE_NOT_FOUND"

    def test_validate_language_override_valid_codes(self, transcriber):
        """Test validating valid language codes."""
        assert transcriber.validate_language_override("en") == "en"
        assert transcriber.validate_language_override("EN") == "en"
        assert transcriber.validate_language_override(" es ") == "es"

    def test_validate_language_override_language_names(self, transcriber):
        """Test validating language names."""
        assert transcriber.validate_language_override("english") == "en"
        assert transcriber.validate_language_override("Spanish") == "es"
        assert transcriber.validate_language_override("FRENCH") == "fr"

    def test_validate_language_override_invalid(self, transcriber):
        """Test validating invalid language."""
        with pytest.raises(ValueError) as exc_info:
            transcriber.validate_language_override("invalid")

        assert "not supported" in str(exc_info.value)

    def test_validate_language_override_empty(self, transcriber):
        """Test validating empty language."""
        with pytest.raises(ValueError) as exc_info:
            transcriber.validate_language_override("")

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_language_override_close_match(self, transcriber):
        """Test validation with close matches providing suggestions."""
        with pytest.raises(ValueError) as exc_info:
            transcriber.validate_language_override("englis")  # typo in "english"

        error_msg = str(exc_info.value)
        assert "Did you mean" in error_msg
        assert "english" in error_msg

    def test_get_language_name(self, transcriber):
        """Test getting language names from codes."""
        assert transcriber.get_language_name("en") == "English"
        assert transcriber.get_language_name("es") == "Spanish"
        assert transcriber.get_language_name("fr") == "French"
        assert transcriber.get_language_name("unknown") == "UNKNOWN"

    def test_get_supported_languages(self, transcriber):
        """Test getting supported languages list."""
        languages = transcriber.get_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 70  # Whisper supports many languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages


class TestEnhancedTranscription:
    """Test enhanced transcription with language detection."""

    @patch("whisper.load_model")
    def test_transcribe_with_manual_language_override(
        self, mock_load_model, transcriber, mock_audio_info, mock_whisper_result
    ):
        """Test transcription with manual language override."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(mock_audio_info, language="english")

        assert isinstance(result, TranscriptionResult)
        assert result.language_detection is not None
        assert result.language_detection.detection_method == "manual"
        assert result.language_detection.detected_language == "en"
        assert result.language_detection.confidence == 1.0

    def test_transcribe_with_invalid_language_override(
        self, transcriber, mock_audio_info
    ):
        """Test transcription with invalid language override."""
        with pytest.raises(AudioProcessingError) as exc_info:
            transcriber.transcribe_audio(mock_audio_info, language="invalid_language")

        assert exc_info.value.error_code.value == "INVALID_INPUT"
        assert "Invalid language specification" in str(exc_info.value)

    @patch("whisper.load_model")
    @patch.object(WhisperTranscriber, "detect_language")
    def test_transcribe_with_auto_detection(
        self,
        mock_detect,
        mock_load_model,
        transcriber,
        mock_audio_info,
        mock_whisper_result,
    ):
        """Test transcription with automatic language detection."""
        # Mock language detection
        mock_detect.return_value = LanguageDetectionResult(
            detected_language="es",
            confidence=0.85,
            all_probabilities={"es": 0.85, "en": 0.12, "fr": 0.03},
            detection_method="whisper",
        )

        # Mock transcription - modify result to match detected language
        whisper_result_with_es = mock_whisper_result.copy()
        whisper_result_with_es["language"] = "es"
        whisper_result_with_es["language_probability"] = 0.85

        mock_model = MagicMock()
        mock_model.transcribe.return_value = whisper_result_with_es
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(
            mock_audio_info, auto_detect_language=True, min_detection_confidence=0.8
        )

        assert isinstance(result, TranscriptionResult)
        assert result.language_detection.detected_language == "es"
        mock_detect.assert_called_once()

    @patch("whisper.load_model")
    @patch.object(WhisperTranscriber, "detect_language")
    def test_transcribe_with_low_confidence_detection(
        self,
        mock_detect,
        mock_load_model,
        transcriber,
        mock_audio_info,
        mock_whisper_result,
    ):
        """Test transcription with low confidence auto-detection."""
        # Mock low confidence detection
        mock_detect.return_value = LanguageDetectionResult(
            detected_language="es",
            confidence=0.4,  # Below threshold
            all_probabilities={"es": 0.4, "en": 0.35, "fr": 0.25},
            detection_method="whisper",
        )

        # Mock transcription
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(
            mock_audio_info, auto_detect_language=True, min_detection_confidence=0.5
        )

        # Should proceed with None language (auto-detection during transcription)
        assert isinstance(result, TranscriptionResult)
        mock_detect.assert_called_once()

    @patch("whisper.load_model")
    @patch.object(WhisperTranscriber, "detect_language")
    def test_transcribe_with_detection_failure(
        self,
        mock_detect,
        mock_load_model,
        transcriber,
        mock_audio_info,
        mock_whisper_result,
    ):
        """Test transcription when language detection fails."""
        # Mock detection failure
        mock_detect.side_effect = AudioProcessingError(
            "Detection failed", ErrorCode.AUDIO_EXTRACTION_FAILED
        )

        # Mock transcription
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        result = transcriber.transcribe_audio(
            mock_audio_info, auto_detect_language=True
        )

        # Should still succeed with fallback
        assert isinstance(result, TranscriptionResult)
        mock_detect.assert_called_once()


class TestTranscriberIntegration:
    """Test integration scenarios for transcriber."""

    @patch("whisper.load_model")
    def test_full_transcription_workflow(
        self, mock_load_model, mock_audio_info, mock_whisper_result
    ):
        """Test complete transcription workflow."""
        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = mock_whisper_result
        mock_load_model.return_value = mock_model

        # Create transcriber and run transcription
        transcriber = create_transcriber()
        result = transcriber.transcribe_audio(mock_audio_info)

        # Verify complete result
        assert isinstance(result, TranscriptionResult)
        assert result.text
        assert len(result.segments) > 0
        assert result.word_count > 0
        assert result.processing_time > 0
        assert result.language_detection is not None

        # Test result methods
        text_segment = result.get_text_between_times(0.0, 2.0)
        assert text_segment

        wpm = result.get_speaking_rate()
        assert wpm >= 0

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        # Cleanup
        transcriber.cleanup()

    @patch("whisper.load_model")
    def test_enhanced_transcription_with_language_features(
        self, mock_load_model, mock_audio_info, mock_whisper_result
    ):
        """Test transcription with all language enhancement features."""
        # Mock transcription result with Spanish
        spanish_result = mock_whisper_result.copy()
        spanish_result["language"] = "es"
        spanish_result["language_probability"] = 0.95

        mock_model = MagicMock()
        mock_model.transcribe.return_value = spanish_result
        mock_load_model.return_value = mock_model

        transcriber = create_transcriber()

        # Test with manual override
        result1 = transcriber.transcribe_audio(mock_audio_info, language="spanish")
        assert result1.language_detection.detection_method == "manual"
        assert result1.language_detection.detected_language == "es"

        # Test language name resolution
        assert transcriber.get_language_name("es") == "Spanish"

        # Test language validation
        normalized = transcriber.validate_language_override("Spanish")
        assert normalized == "es"

        # Cleanup
        transcriber.cleanup()

    def test_error_recovery_scenarios(self, transcriber):
        """Test error recovery in various scenarios."""
        # Test that transcriber can handle various error conditions
        # This would be expanded with specific error scenarios
        pass
