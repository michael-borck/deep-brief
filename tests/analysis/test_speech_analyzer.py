"""Tests for speech analysis functionality."""

import pytest

from deep_brief.analysis.speech_analyzer import (
    OverallSpeechMetrics,
    SceneSpeechMetrics,
    SpeechAnalysisResult,
    SpeechAnalyzer,
    create_speech_analyzer,
)
from deep_brief.analysis.transcriber import (
    Segment,
    TranscriptionResult,
    WordTimestamp,
)
from deep_brief.core.scene_detector import Scene, SceneDetectionResult
from deep_brief.utils.config import AnalysisConfig, DeepBriefConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = DeepBriefConfig(
        analysis=AnalysisConfig(
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )
    )
    return config


@pytest.fixture
def speech_analyzer(mock_config):
    """Create SpeechAnalyzer instance for testing."""
    return SpeechAnalyzer(config=mock_config)


@pytest.fixture
def sample_transcription_result():
    """Create sample transcription result for testing."""
    segments = [
        Segment(
            id=0,
            text="Hello world. This is scene one.",
            start=0.0,
            end=5.0,
            avg_logprob=-0.3,
            no_speech_prob=0.1,
            words=[
                WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.95),
                WordTimestamp(word="world", start=0.6, end=1.2, confidence=0.92),
                WordTimestamp(word=".", start=1.2, end=1.3, confidence=0.99),
                WordTimestamp(word="This", start=2.0, end=2.3, confidence=0.88),
                WordTimestamp(word="is", start=2.4, end=2.6, confidence=0.94),
                WordTimestamp(word="scene", start=2.7, end=3.2, confidence=0.91),
                WordTimestamp(word="one", start=3.3, end=3.7, confidence=0.89),
                WordTimestamp(word=".", start=3.7, end=3.8, confidence=0.99),
            ],
            language="en",
        ),
        Segment(
            id=1,
            text="Welcome to scene two. This is faster speech.",
            start=10.0,
            end=13.0,
            avg_logprob=-0.2,
            no_speech_prob=0.05,
            words=[
                WordTimestamp(word="Welcome", start=10.0, end=10.5, confidence=0.96),
                WordTimestamp(word="to", start=10.6, end=10.8, confidence=0.98),
                WordTimestamp(word="scene", start=10.9, end=11.3, confidence=0.93),
                WordTimestamp(word="two", start=11.4, end=11.7, confidence=0.90),
                WordTimestamp(word=".", start=11.7, end=11.8, confidence=0.99),
                WordTimestamp(word="This", start=12.0, end=12.2, confidence=0.87),
                WordTimestamp(word="is", start=12.3, end=12.4, confidence=0.95),
                WordTimestamp(word="faster", start=12.5, end=12.8, confidence=0.92),
                WordTimestamp(word="speech", start=12.9, end=13.2, confidence=0.88),
                WordTimestamp(word=".", start=13.2, end=13.3, confidence=0.99),
            ],
            language="en",
        ),
    ]

    return TranscriptionResult(
        text="Hello world. This is scene one. Welcome to scene two. This is faster speech.",
        segments=segments,
        language="en",
        language_probability=0.95,
        duration=20.0,
        model_used="whisper-base",
        word_count=18,
        processing_time=2.5,
    )


@pytest.fixture
def sample_scene_result():
    """Create sample scene detection result for testing."""
    scenes = [
        Scene(
            start_time=0.0,
            end_time=8.0,
            duration=8.0,
            scene_number=1,
            confidence=0.8,
        ),
        Scene(
            start_time=8.0,
            end_time=15.0,
            duration=7.0,
            scene_number=2,
            confidence=0.7,
        ),
    ]

    return SceneDetectionResult(
        scenes=scenes,
        total_scenes=2,
        detection_method="threshold",
        threshold_used=0.4,
        video_duration=20.0,
        average_scene_duration=7.5,
    )


class TestSceneSpeechMetrics:
    """Test SceneSpeechMetrics model."""

    def test_scene_speech_metrics_creation(self):
        """Test creating SceneSpeechMetrics."""
        metrics = SceneSpeechMetrics(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            word_count=20,
            speaking_rate_wpm=120.0,
            text_content="Hello world. This is a test.",
            character_count=28,
            sentence_count=2,
            actual_speaking_time=8.0,
            pause_time=2.0,
            speech_ratio=0.8,
            avg_confidence=0.95,
            low_confidence_words=1,
            filler_word_count=2,
            filler_word_rate=12.0,
            filler_words_detected=[
                {
                    "word": "um",
                    "normalized": "um",
                    "start": 1.0,
                    "end": 1.2,
                    "confidence": 0.9,
                },
                {
                    "word": "like",
                    "normalized": "like",
                    "start": 5.0,
                    "end": 5.3,
                    "confidence": 0.85,
                },
            ],
            silence_segments=[
                {"start": 2.0, "end": 3.0, "duration": 1.0, "type": "inter_word"}
            ],
            total_silence_time=1.0,
            silence_ratio=0.1,
            average_silence_duration=1.0,
            longest_silence_duration=1.0,
            silence_count=1,
            sentiment_score=0.0,
            sentiment_label="neutral",
            sentiment_confidence=0.5,
            emotional_keywords=[],
            quality_score=0.85,
            quality_category="good",
            confidence_issues=[],
            segment_confidence_scores=[],
            avg_segment_logprob=-0.3,
            no_speech_ratio=0.1,
            problematic_segments=[],
        )

        assert metrics.scene_number == 1
        assert metrics.duration == 10.0
        assert metrics.speaking_rate_wpm == 120.0
        assert metrics.speech_ratio == 0.8
        assert metrics.avg_confidence == 0.95


class TestOverallSpeechMetrics:
    """Test OverallSpeechMetrics model."""

    def test_overall_speech_metrics_creation(self):
        """Test creating OverallSpeechMetrics."""
        metrics = OverallSpeechMetrics(
            total_duration=20.0,
            total_scenes=2,
            total_word_count=40,
            average_wpm=150.0,
            median_wpm=145.0,
            min_wpm=120.0,
            max_wpm=180.0,
            overall_speech_ratio=0.75,
            total_speaking_time=15.0,
            total_pause_time=5.0,
            wpm_variance=400.0,
            wpm_standard_deviation=20.0,
            total_characters=200,
            total_sentences=8,
            average_words_per_sentence=5.0,
            overall_confidence=0.92,
            low_confidence_percentage=5.0,
            total_filler_words=5,
            filler_word_rate=15.0,
            most_common_filler_words=[
                {"word": "um", "count": 3},
                {"word": "like", "count": 2},
            ],
            total_silence_time=5.0,
            overall_silence_ratio=0.25,
            average_silence_per_scene=2.5,
            longest_silence_overall=3.0,
            total_silence_segments=4,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 2, "fair": 0, "poor": 0},
            average_confidence=0.92,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        assert metrics.total_duration == 20.0
        assert metrics.total_scenes == 2
        assert metrics.average_wpm == 150.0
        assert metrics.overall_speech_ratio == 0.75


class TestSpeechAnalysisResult:
    """Test SpeechAnalysisResult model and methods."""

    def test_speech_analysis_result_creation(self):
        """Test creating SpeechAnalysisResult."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,
                text_content="Scene one content",
                character_count=17,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            )
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=10.0,
            total_scenes=1,
            total_word_count=20,
            average_wpm=120.0,
            median_wpm=120.0,
            min_wpm=120.0,
            max_wpm=120.0,
            overall_speech_ratio=0.8,
            total_speaking_time=8.0,
            total_pause_time=2.0,
            wpm_variance=0.0,
            wpm_standard_deviation=0.0,
            total_characters=17,
            total_sentences=1,
            average_words_per_sentence=20.0,
            overall_confidence=0.95,
            low_confidence_percentage=0.0,
            total_filler_words=0,
            filler_word_rate=0.0,
            most_common_filler_words=[],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 1},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 1, "fair": 0, "poor": 0},
            average_confidence=0.95,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        assert len(result.scene_metrics) == 1
        assert result.language == "en"
        assert result.target_wpm_range == [140, 160]

    def test_get_scene_metrics(self):
        """Test getting metrics for specific scene."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,
                text_content="Scene one",
                character_count=9,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=30,
                speaking_rate_wpm=180.0,
                text_content="Scene two",
                character_count=9,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.92,
                low_confidence_words=1,
                filler_word_count=1,
                filler_word_rate=6.0,
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 11.0,
                        "end": 11.2,
                        "confidence": 0.9,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=20.0,
            total_scenes=2,
            total_word_count=50,
            average_wpm=150.0,
            median_wpm=150.0,
            min_wpm=120.0,
            max_wpm=180.0,
            overall_speech_ratio=0.85,
            total_speaking_time=17.0,
            total_pause_time=3.0,
            wpm_variance=100.0,
            wpm_standard_deviation=10.0,
            total_characters=200,
            total_sentences=5,
            average_words_per_sentence=10.0,
            overall_confidence=0.90,
            low_confidence_percentage=5.0,
            total_filler_words=3,
            filler_word_rate=9.0,
            most_common_filler_words=[
                {"word": "um", "count": 2},
                {"word": "like", "count": 1},
            ],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            average_confidence=0.75,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test existing scene
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None
        assert scene_1.scene_number == 1
        assert scene_1.speaking_rate_wpm == 120.0

        # Test non-existent scene
        scene_3 = result.get_scene_metrics(3)
        assert scene_3 is None

    def test_get_scenes_by_wpm_range(self):
        """Test filtering scenes by WPM range."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,  # Below range
                text_content="Slow scene",
                character_count=10,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=25,
                speaking_rate_wpm=150.0,  # In range
                text_content="Normal scene",
                character_count=12,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.92,
                low_confidence_words=1,
                filler_word_count=1,
                filler_word_rate=6.0,
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 11.0,
                        "end": 11.2,
                        "confidence": 0.9,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=3,
                start_time=20.0,
                end_time=30.0,
                duration=10.0,
                word_count=35,
                speaking_rate_wpm=200.0,  # Above range
                text_content="Fast scene",
                character_count=10,
                sentence_count=1,
                actual_speaking_time=9.5,
                pause_time=0.5,
                speech_ratio=0.95,
                avg_confidence=0.88,
                low_confidence_words=2,
                filler_word_count=2,
                filler_word_rate=12.0,
                filler_words_detected=[
                    {
                        "word": "like",
                        "normalized": "like",
                        "start": 21.0,
                        "end": 21.3,
                        "confidence": 0.88,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=30.0,
            total_scenes=3,
            total_word_count=80,
            average_wpm=160.0,
            median_wpm=150.0,
            min_wpm=120.0,
            max_wpm=200.0,
            overall_speech_ratio=0.88,
            total_speaking_time=26.5,
            total_pause_time=3.5,
            wpm_variance=1600.0,
            wpm_standard_deviation=40.0,
            total_characters=320,
            total_sentences=3,
            average_words_per_sentence=26.7,
            overall_confidence=0.88,
            low_confidence_percentage=10.0,
            total_filler_words=5,
            filler_word_rate=10.0,
            most_common_filler_words=[
                {"word": "like", "count": 3},
                {"word": "um", "count": 2},
            ],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 3},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            average_confidence=0.75,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test range filtering
        in_range = result.get_scenes_by_wpm_range(140, 160)
        assert len(in_range) == 1
        assert in_range[0].scene_number == 2

        # Test slow scenes
        slow_scenes = result.get_slow_scenes()
        assert len(slow_scenes) == 1
        assert slow_scenes[0].scene_number == 1

        # Test fast scenes
        fast_scenes = result.get_fast_scenes()
        assert len(fast_scenes) == 1
        assert fast_scenes[0].scene_number == 3

        # Test optimal scenes
        optimal_scenes = result.get_optimal_scenes()
        assert len(optimal_scenes) == 1
        assert optimal_scenes[0].scene_number == 2

    def test_to_dict(self):
        """Test converting result to dictionary."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=150.0,
                text_content="Test scene",
                character_count=10,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=1,
                filler_word_rate=6.0,
                filler_words_detected=[
                    {
                        "word": "like",
                        "normalized": "like",
                        "start": 5.0,
                        "end": 5.3,
                        "confidence": 0.9,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            )
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=10.0,
            total_scenes=1,
            total_word_count=20,
            average_wpm=150.0,
            median_wpm=150.0,
            min_wpm=150.0,
            max_wpm=150.0,
            overall_speech_ratio=0.8,
            total_speaking_time=8.0,
            total_pause_time=2.0,
            wpm_variance=0.0,
            wpm_standard_deviation=0.0,
            total_characters=10,
            total_sentences=1,
            average_words_per_sentence=20.0,
            overall_confidence=0.95,
            low_confidence_percentage=0.0,
            total_filler_words=1,
            filler_word_rate=6.0,
            most_common_filler_words=[{"word": "like", "count": 1}],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 1},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 1, "fair": 0, "poor": 0},
            average_confidence=0.95,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        result_dict = result.to_dict()

        assert "scene_metrics" in result_dict
        assert "overall_metrics" in result_dict
        assert "analysis_summary" in result_dict
        assert result_dict["language"] == "en"
        assert result_dict["target_wpm_range"] == [140, 160]

        # Test analysis summary
        summary = result_dict["analysis_summary"]
        assert summary["total_scenes"] == 1
        assert summary["optimal_scenes"] == 1
        assert summary["slow_scenes"] == 0
        assert summary["fast_scenes"] == 0


class TestSpeechAnalyzer:
    """Test SpeechAnalyzer class."""

    def test_initialization_default_config(self):
        """Test analyzer initialization with default config."""
        analyzer = SpeechAnalyzer()
        assert analyzer.config is not None

    def test_initialization_custom_config(self, mock_config):
        """Test analyzer initialization with custom config."""
        analyzer = SpeechAnalyzer(config=mock_config)
        assert analyzer.config == mock_config

    def test_analyze_speech_success(
        self, speech_analyzer, sample_transcription_result, sample_scene_result
    ):
        """Test successful speech analysis."""
        result = speech_analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result
        )

        assert isinstance(result, SpeechAnalysisResult)
        assert len(result.scene_metrics) == 2
        assert result.overall_metrics.total_scenes == 2
        assert result.language == "en"
        assert result.model_used == "whisper-base"
        assert result.processing_time > 0

        # Verify scene metrics
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None
        assert scene_1.start_time == 0.0
        assert scene_1.end_time == 8.0
        assert scene_1.word_count > 0

        scene_2 = result.get_scene_metrics(2)
        assert scene_2 is not None
        assert scene_2.start_time == 8.0
        assert scene_2.end_time == 15.0

    def test_analyze_speech_with_custom_confidence(
        self, speech_analyzer, sample_transcription_result, sample_scene_result
    ):
        """Test speech analysis with custom confidence threshold."""
        result = speech_analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result, confidence_threshold=0.9
        )

        assert result.confidence_threshold == 0.9
        assert isinstance(result, SpeechAnalysisResult)

    def test_get_scene_words_filtering(
        self, speech_analyzer, sample_transcription_result
    ):
        """Test word filtering by scene timeframe and confidence."""
        # Test with normal confidence threshold
        words = speech_analyzer._get_scene_words(
            sample_transcription_result, 0.0, 5.0, 0.7
        )

        # Should include words from first segment that meet confidence threshold
        assert len(words) > 0
        for word in words:
            assert word["confidence"] >= 0.7
            assert 0.0 <= word["start"] <= 5.0
            assert 0.0 <= word["end"] <= 5.0

        # Test with high confidence threshold
        high_conf_words = speech_analyzer._get_scene_words(
            sample_transcription_result, 0.0, 5.0, 0.95
        )

        # Should have fewer words with higher threshold
        assert len(high_conf_words) <= len(words)

    def test_calculate_actual_speaking_time(self, speech_analyzer):
        """Test calculation of actual speaking time."""
        scene_words = [
            {
                "start": 1.0,
                "end": 1.5,
                "word": "hello",
                "confidence": 0.9,
                "duration": 0.5,
            },
            {
                "start": 2.0,
                "end": 2.3,
                "word": "world",
                "confidence": 0.8,
                "duration": 0.3,
            },
            {
                "start": 3.0,
                "end": 3.2,
                "word": "test",
                "confidence": 0.95,
                "duration": 0.2,
            },
        ]

        speaking_time = speech_analyzer._calculate_actual_speaking_time(
            scene_words, 0.0, 5.0
        )

        # Should sum up the duration of all words
        expected_time = 0.5 + 0.3 + 0.2  # 1.0 seconds
        assert speaking_time == expected_time

        # Test with no words
        no_words_time = speech_analyzer._calculate_actual_speaking_time([], 0.0, 5.0)
        assert no_words_time == 0.0

    def test_count_sentences(self, speech_analyzer):
        """Test sentence counting."""
        # Test various sentence patterns
        assert speech_analyzer._count_sentences("Hello world.") == 1
        assert speech_analyzer._count_sentences("Hello! How are you?") == 2
        assert speech_analyzer._count_sentences("One. Two! Three?") == 3
        assert speech_analyzer._count_sentences("No punctuation") == 1
        assert speech_analyzer._count_sentences("") == 0
        assert speech_analyzer._count_sentences("   ") == 0

    def test_calculate_confidence_metrics(self, speech_analyzer):
        """Test confidence metrics calculation."""
        scene_words = [
            {"confidence": 0.95},
            {"confidence": 0.85},
            {"confidence": 0.65},  # Below threshold
            {"confidence": 0.75},
            {"confidence": 0.60},  # Below threshold
        ]

        avg_conf, low_conf_count = speech_analyzer._calculate_confidence_metrics(
            scene_words, 0.7
        )

        expected_avg = (0.95 + 0.85 + 0.65 + 0.75 + 0.60) / 5
        assert avg_conf == expected_avg
        assert low_conf_count == 2  # Two words below 0.7

        # Test with no words
        avg_empty, low_empty = speech_analyzer._calculate_confidence_metrics([], 0.7)
        assert avg_empty == 0.0
        assert low_empty == 0

    def test_calculate_overall_metrics_empty(self, speech_analyzer):
        """Test overall metrics calculation with no scenes."""
        metrics = speech_analyzer._calculate_overall_metrics([], 10.0)

        assert metrics.total_duration == 10.0
        assert metrics.total_scenes == 0
        assert metrics.total_word_count == 0
        assert metrics.average_wpm == 0.0
        assert metrics.overall_speech_ratio == 0.0

    def test_calculate_overall_metrics_with_scenes(self, speech_analyzer):
        """Test overall metrics calculation with scene data."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,
                text_content="Scene one content",
                character_count=17,
                sentence_count=2,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=1,
                filler_word_count=1,
                filler_word_rate=6.0,
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 1.0,
                        "end": 1.2,
                        "confidence": 0.95,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=30,
                speaking_rate_wpm=180.0,
                text_content="Scene two content",
                character_count=17,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.90,
                low_confidence_words=2,
                filler_word_count=2,
                filler_word_rate=12.0,
                filler_words_detected=[
                    {
                        "word": "like",
                        "normalized": "like",
                        "start": 11.0,
                        "end": 11.3,
                        "confidence": 0.90,
                    },
                    {
                        "word": "uh",
                        "normalized": "uh",
                        "start": 12.0,
                        "end": 12.2,
                        "confidence": 0.88,
                    },
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        metrics = speech_analyzer._calculate_overall_metrics(scene_metrics, 20.0)

        assert metrics.total_duration == 20.0
        assert metrics.total_scenes == 2
        assert metrics.total_word_count == 50
        assert metrics.average_wpm == 150.0  # (120 + 180) / 2
        assert metrics.median_wpm == 150.0  # Median of [120, 180]
        assert metrics.min_wpm == 120.0
        assert metrics.max_wpm == 180.0
        assert metrics.total_speaking_time == 17.0  # 8 + 9
        assert metrics.total_pause_time == 3.0  # 2 + 1
        assert metrics.total_characters == 34  # 17 + 17
        assert metrics.total_sentences == 3  # 2 + 1

        # Test weighted confidence calculation
        expected_confidence = (0.95 * 20 + 0.90 * 30) / 50  # Weighted by word count
        assert abs(metrics.overall_confidence - expected_confidence) < 0.01

        # Test low confidence percentage
        expected_low_conf_pct = (3 / 50) * 100  # 3 low confidence words out of 50
        assert abs(metrics.low_confidence_percentage - expected_low_conf_pct) < 0.01


class TestSpeechAnalyzerFactory:
    """Test speech analyzer factory function."""

    def test_create_speech_analyzer_no_config(self):
        """Test creating analyzer without config."""
        analyzer = create_speech_analyzer()

        assert isinstance(analyzer, SpeechAnalyzer)
        assert analyzer.config is not None

    def test_create_speech_analyzer_with_config(self, mock_config):
        """Test creating analyzer with config."""
        analyzer = create_speech_analyzer(config=mock_config)

        assert isinstance(analyzer, SpeechAnalyzer)
        assert analyzer.config == mock_config


class TestFillerWordDetection:
    """Test filler word detection functionality."""

    def test_detect_filler_words_basic(self, speech_analyzer):
        """Test basic filler word detection."""
        scene_words = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {
                "word": "um,",
                "start": 0.6,
                "end": 0.8,
                "confidence": 0.85,
            },  # Filler with punctuation
            {"word": "this", "start": 1.0, "end": 1.3, "confidence": 0.92},
            {"word": "is", "start": 1.4, "end": 1.6, "confidence": 0.94},
            {"word": "like", "start": 1.7, "end": 2.0, "confidence": 0.88},  # Filler
            {"word": "a", "start": 2.1, "end": 2.2, "confidence": 0.95},
            {"word": "test", "start": 2.3, "end": 2.7, "confidence": 0.91},
        ]

        duration = 10.0  # 10 seconds
        filler_count, filler_rate, detected_fillers = (
            speech_analyzer._detect_filler_words(scene_words, duration)
        )

        assert filler_count == 2  # "um" and "like"
        assert (
            filler_rate == 12.0
        )  # 2 fillers / (10/60) minutes = 12 fillers per minute
        assert len(detected_fillers) == 2

        # Check detected filler details
        filler_words = [f["normalized"] for f in detected_fillers]
        assert "um" in filler_words
        assert "like" in filler_words

        # Check filler details
        um_filler = next(f for f in detected_fillers if f["normalized"] == "um")
        assert um_filler["word"] == "um,"
        assert um_filler["start"] == 0.6
        assert um_filler["confidence"] == 0.85

    def test_detect_filler_words_empty(self, speech_analyzer):
        """Test filler word detection with no words."""
        filler_count, filler_rate, detected_fillers = (
            speech_analyzer._detect_filler_words([], 10.0)
        )

        assert filler_count == 0
        assert filler_rate == 0.0
        assert detected_fillers == []

    def test_detect_filler_words_case_insensitive(self, speech_analyzer):
        """Test that filler word detection is case insensitive."""
        scene_words = [
            {"word": "UM", "start": 0.0, "end": 0.3, "confidence": 0.9},
            {"word": "Like", "start": 1.0, "end": 1.3, "confidence": 0.88},
            {"word": "UH", "start": 2.0, "end": 2.2, "confidence": 0.85},
        ]

        duration = 3.0
        filler_count, filler_rate, detected_fillers = (
            speech_analyzer._detect_filler_words(scene_words, duration)
        )

        assert filler_count == 3
        assert len(detected_fillers) == 3

        normalized_words = [f["normalized"] for f in detected_fillers]
        assert "um" in normalized_words
        assert "like" in normalized_words
        assert "uh" in normalized_words

    def test_scene_analysis_with_filler_words(
        self, speech_analyzer, sample_transcription_result, sample_scene_result
    ):
        """Test that scene analysis includes filler word metrics."""
        # First, let's modify the transcription to include filler words
        # Add some filler words to the existing segments
        sample_transcription_result.segments[0].words.extend(
            [
                WordTimestamp(word="um", start=4.0, end=4.2, confidence=0.85),
                WordTimestamp(word="like", start=4.5, end=4.8, confidence=0.90),
            ]
        )

        result = speech_analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result
        )

        # Check that filler word metrics are included
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None
        assert hasattr(scene_1, "filler_word_count")
        assert hasattr(scene_1, "filler_word_rate")
        assert hasattr(scene_1, "filler_words_detected")

        # Check overall metrics include filler word data
        assert hasattr(result.overall_metrics, "total_filler_words")
        assert hasattr(result.overall_metrics, "filler_word_rate")
        assert hasattr(result.overall_metrics, "most_common_filler_words")

    def test_filler_word_filtering_methods(self, speech_analyzer):
        """Test scene filtering by filler word metrics."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=150.0,
                text_content="Low filler scene",
                character_count=16,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=1,
                filler_word_rate=1.5,  # Low filler rate
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 1.0,
                        "end": 1.2,
                        "confidence": 0.9,
                    }
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=25,
                speaking_rate_wpm=160.0,
                text_content="High filler scene",
                character_count=17,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.92,
                low_confidence_words=1,
                filler_word_count=8,
                filler_word_rate=8.0,  # High filler rate
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 11.0,
                        "end": 11.2,
                        "confidence": 0.9,
                    },
                    {
                        "word": "like",
                        "normalized": "like",
                        "start": 12.0,
                        "end": 12.3,
                        "confidence": 0.88,
                    },
                ],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=20.0,
            total_scenes=2,
            total_word_count=45,
            average_wpm=155.0,
            median_wpm=155.0,
            min_wpm=150.0,
            max_wpm=160.0,
            overall_speech_ratio=0.85,
            total_speaking_time=17.0,
            total_pause_time=3.0,
            wpm_variance=25.0,
            wpm_standard_deviation=5.0,
            total_characters=33,
            total_sentences=2,
            average_words_per_sentence=22.5,
            overall_confidence=0.935,
            low_confidence_percentage=2.2,
            total_filler_words=9,
            filler_word_rate=4.5,
            most_common_filler_words=[
                {"word": "um", "count": 5},
                {"word": "like", "count": 4},
            ],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            average_confidence=0.75,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test filler word filtering
        high_filler_scenes = result.get_scenes_with_high_filler_words(
            threshold_rate=5.0
        )
        assert len(high_filler_scenes) == 1
        assert high_filler_scenes[0].scene_number == 2

        low_filler_scenes = result.get_scenes_with_low_filler_words(threshold_rate=2.0)
        assert len(low_filler_scenes) == 1
        assert low_filler_scenes[0].scene_number == 1

        # Test filler word aggregation
        filler_counts = result.get_total_filler_words_by_type()
        assert "um" in filler_counts
        assert "like" in filler_counts


class TestSilenceDetection:
    """Test silence detection functionality."""

    def test_detect_silence_segments_basic(self, speech_analyzer):
        """Test basic silence detection between words."""
        scene_words = [
            {"start": 1.0, "end": 1.5, "word": "Hello"},
            {"start": 3.0, "end": 3.5, "word": "world"},  # 1.5s gap before this word
            {"start": 4.0, "end": 4.5, "word": "test"},  # 0.5s gap (will be detected)
        ]

        start_time = 0.0
        end_time = 6.0
        min_silence_duration = 0.5

        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = speech_analyzer._detect_silence_segments(
            scene_words, start_time, end_time, min_silence_duration
        )

        # Should detect: pre-speech (0-1), inter-word (1.5-3), inter-word (3.5-4), post-speech (4.5-6)
        assert silence_count == 4
        assert len(silence_segments) == 4

        # Check total silence time: 1.0 + 1.5 + 0.5 + 1.5 = 4.5 seconds
        assert total_silence_time == 4.5
        assert silence_ratio == 0.75  # 4.5/6.0
        assert average_silence_duration == 1.125  # 4.5/4
        assert longest_silence_duration == 1.5  # Two segments of 1.5s each

        # Check segment types
        segment_types = [seg["type"] for seg in silence_segments]
        assert "pre_speech" in segment_types
        assert "inter_word" in segment_types
        assert "post_speech" in segment_types

    def test_detect_silence_segments_empty_scene(self, speech_analyzer):
        """Test silence detection with no words (entire scene is silence)."""
        scene_words = []
        start_time = 0.0
        end_time = 10.0

        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = speech_analyzer._detect_silence_segments(scene_words, start_time, end_time)

        assert silence_count == 1
        assert len(silence_segments) == 1
        assert silence_segments[0]["type"] == "scene_silence"
        assert total_silence_time == 10.0
        assert silence_ratio == 1.0
        assert average_silence_duration == 10.0
        assert longest_silence_duration == 10.0

    def test_detect_silence_segments_no_silence(self, speech_analyzer):
        """Test scene with continuous speech (no detectable silence)."""
        scene_words = [
            {"start": 0.0, "end": 0.5, "word": "Hello"},
            {"start": 0.5, "end": 1.0, "word": "world"},  # No gap
            {"start": 1.0, "end": 1.5, "word": "test"},  # No gap
        ]

        start_time = 0.0
        end_time = 1.5
        min_silence_duration = 0.5

        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = speech_analyzer._detect_silence_segments(
            scene_words, start_time, end_time, min_silence_duration
        )

        assert silence_count == 0
        assert len(silence_segments) == 0
        assert total_silence_time == 0.0
        assert silence_ratio == 0.0
        assert average_silence_duration == 0.0
        assert longest_silence_duration == 0.0

    def test_detect_silence_segments_custom_threshold(self, speech_analyzer):
        """Test silence detection with custom minimum duration threshold."""
        scene_words = [
            {"start": 1.0, "end": 1.5, "word": "Hello"},
            {
                "start": 1.8,
                "end": 2.0,
                "word": "world",
            },  # 0.3s gap (below 0.5s threshold)
            {"start": 3.0, "end": 3.5, "word": "test"},  # 1.0s gap (above threshold)
        ]

        start_time = 0.0
        end_time = 4.0
        min_silence_duration = 0.5

        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = speech_analyzer._detect_silence_segments(
            scene_words, start_time, end_time, min_silence_duration
        )

        # Should detect: pre-speech (0-1), inter-word (2-3)
        # The 0.3s gap should be ignored (below threshold)
        # Post-speech (3.5-4) is 0.5s but condition requires > min_silence_duration
        assert silence_count == 2  # pre-speech (1.0s) and inter-word (1.0s) only
        assert total_silence_time == 2.0  # 1.0 + 1.0

        # Test with lower threshold
        min_silence_duration = 0.2

        (
            silence_segments_lower,
            total_silence_time_lower,
            silence_ratio_lower,
            average_silence_duration_lower,
            longest_silence_duration_lower,
            silence_count_lower,
        ) = speech_analyzer._detect_silence_segments(
            scene_words, start_time, end_time, min_silence_duration
        )

        # Now should detect the 0.3s gap and post-speech as well
        assert silence_count_lower == 4
        assert total_silence_time_lower == 2.8  # 1.0 + 0.3 + 1.0 + 0.5

    def test_scene_analysis_with_silence_integration(
        self, speech_analyzer, sample_transcription_result, sample_scene_result
    ):
        """Test that scene analysis includes silence metrics."""
        result = speech_analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result
        )

        # Check that silence metrics are included in scene results
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None
        assert hasattr(scene_1, "silence_segments")
        assert hasattr(scene_1, "total_silence_time")
        assert hasattr(scene_1, "silence_ratio")
        assert hasattr(scene_1, "average_silence_duration")
        assert hasattr(scene_1, "longest_silence_duration")
        assert hasattr(scene_1, "silence_count")

        # Check overall metrics include silence data
        assert hasattr(result.overall_metrics, "total_silence_time")
        assert hasattr(result.overall_metrics, "overall_silence_ratio")
        assert hasattr(result.overall_metrics, "average_silence_per_scene")
        assert hasattr(result.overall_metrics, "longest_silence_overall")
        assert hasattr(result.overall_metrics, "total_silence_segments")

    def test_silence_filtering_methods(self, speech_analyzer):
        """Test scene filtering by silence metrics."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=150.0,
                text_content="Low silence scene",
                character_count=17,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[
                    {"start": 5.0, "end": 5.5, "duration": 0.5, "type": "inter_word"}
                ],
                total_silence_time=0.5,
                silence_ratio=0.05,  # Low silence ratio
                average_silence_duration=0.5,
                longest_silence_duration=0.5,
                silence_count=1,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=15,
                speaking_rate_wpm=120.0,
                text_content="High silence scene",
                character_count=18,
                sentence_count=1,
                actual_speaking_time=6.0,
                pause_time=4.0,
                speech_ratio=0.6,
                avg_confidence=0.92,
                low_confidence_words=1,
                filler_word_count=1,
                filler_word_rate=6.0,
                filler_words_detected=[
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 15.0,
                        "end": 15.2,
                        "confidence": 0.9,
                    }
                ],
                silence_segments=[
                    {"start": 12.0, "end": 14.0, "duration": 2.0, "type": "inter_word"},
                    {"start": 16.0, "end": 18.0, "duration": 2.0, "type": "inter_word"},
                ],
                total_silence_time=4.0,
                silence_ratio=0.4,  # High silence ratio
                average_silence_duration=2.0,
                longest_silence_duration=2.0,
                silence_count=2,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.85,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.3,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=20.0,
            total_scenes=2,
            total_word_count=35,
            average_wpm=135.0,
            median_wpm=135.0,
            min_wpm=120.0,
            max_wpm=150.0,
            overall_speech_ratio=0.75,
            total_speaking_time=15.0,
            total_pause_time=5.0,
            wpm_variance=225.0,
            wpm_standard_deviation=15.0,
            total_characters=35,
            total_sentences=2,
            average_words_per_sentence=17.5,
            overall_confidence=0.935,
            low_confidence_percentage=2.9,
            total_filler_words=1,
            filler_word_rate=3.0,
            most_common_filler_words=[{"word": "um", "count": 1}],
            total_silence_time=4.5,
            overall_silence_ratio=0.225,
            average_silence_per_scene=2.25,
            longest_silence_overall=2.0,
            total_silence_segments=3,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.85,
            quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            average_confidence=0.75,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test silence filtering
        excessive_silence_scenes = result.get_scenes_with_excessive_silence(
            threshold_ratio=0.3
        )
        assert len(excessive_silence_scenes) == 1
        assert excessive_silence_scenes[0].scene_number == 2

        minimal_silence_scenes = result.get_scenes_with_minimal_silence(
            threshold_ratio=0.1
        )
        assert len(minimal_silence_scenes) == 1
        assert minimal_silence_scenes[0].scene_number == 1

        # Test silence segment aggregation
        all_silences = result.get_all_silence_segments()
        assert len(all_silences) == 3

        # Test longest silences
        longest_silences = result.get_longest_silences(count=2)
        assert len(longest_silences) == 2
        assert all(s["duration"] >= 1.0 for s in longest_silences)


class TestSpeechAnalyzerIntegration:
    """Test integration scenarios for speech analyzer."""

    def test_full_analysis_workflow(
        self, mock_config, sample_transcription_result, sample_scene_result
    ):
        """Test complete analysis workflow."""
        analyzer = create_speech_analyzer(config=mock_config)

        result = analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result
        )

        # Verify complete result structure
        assert isinstance(result, SpeechAnalysisResult)
        assert len(result.scene_metrics) == sample_scene_result.total_scenes
        assert result.overall_metrics.total_scenes == sample_scene_result.total_scenes
        assert result.processing_time > 0

        # Test result methods
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "analysis_summary" in result_dict

        # Test scene classification
        slow_scenes = result.get_slow_scenes()
        fast_scenes = result.get_fast_scenes()
        optimal_scenes = result.get_optimal_scenes()

        total_classified = len(slow_scenes) + len(fast_scenes) + len(optimal_scenes)
        assert total_classified == len(result.scene_metrics)

    def test_edge_case_handling(self, speech_analyzer):
        """Test handling of edge cases."""
        # Test with empty transcription (no segments)
        empty_transcription = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            language_probability=0.0,
            duration=10.0,
            model_used="whisper-base",
            word_count=0,
            processing_time=1.0,
        )

        single_scene = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=0.0,
                    end_time=10.0,
                    duration=10.0,
                    scene_number=1,
                    confidence=0.5,
                )
            ],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=10.0,
            average_scene_duration=10.0,
        )

        result = speech_analyzer.analyze_speech(empty_transcription, single_scene)

        # Should handle gracefully
        assert isinstance(result, SpeechAnalysisResult)
        assert len(result.scene_metrics) == 1
        assert result.scene_metrics[0].word_count == 0
        assert result.scene_metrics[0].speaking_rate_wpm == 0.0


class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""

    def test_analyze_sentiment_positive(self, speech_analyzer):
        """Test sentiment analysis with positive text."""
        positive_text = (
            "This is an excellent presentation with great ideas and wonderful insights."
        )

        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            speech_analyzer._analyze_sentiment(positive_text)
        )

        assert sentiment_score > 0
        assert sentiment_label == "positive"
        assert confidence > 0
        assert len(emotional_keywords) > 0

        # Check that emotional keywords are detected
        positive_words_found = [
            kw["word"] for kw in emotional_keywords if kw["sentiment"] == "positive"
        ]
        assert len(positive_words_found) > 0

    def test_analyze_sentiment_negative(self, speech_analyzer):
        """Test sentiment analysis with negative text."""
        negative_text = (
            "This is a terrible presentation with bad ideas and disappointing results."
        )

        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            speech_analyzer._analyze_sentiment(negative_text)
        )

        assert sentiment_score < 0
        assert sentiment_label == "negative"
        assert confidence > 0
        assert len(emotional_keywords) > 0

        # Check that emotional keywords are detected
        negative_words_found = [
            kw["word"] for kw in emotional_keywords if kw["sentiment"] == "negative"
        ]
        assert len(negative_words_found) > 0

    def test_analyze_sentiment_neutral(self, speech_analyzer):
        """Test sentiment analysis with neutral text."""
        neutral_text = (
            "The data shows various statistics and numbers from the quarterly report."
        )

        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            speech_analyzer._analyze_sentiment(neutral_text)
        )

        assert abs(sentiment_score) <= 0.1
        assert sentiment_label == "neutral"
        assert len(emotional_keywords) == 0

    def test_analyze_sentiment_empty_text(self, speech_analyzer):
        """Test sentiment analysis with empty text."""
        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            speech_analyzer._analyze_sentiment("")
        )

        assert sentiment_score == 0.0
        assert sentiment_label == "neutral"
        assert confidence == 0.0
        assert emotional_keywords == []

    def test_analyze_sentiment_mixed(self, speech_analyzer):
        """Test sentiment analysis with mixed positive and negative text."""
        mixed_text = "The presentation has some good points but also terrible mistakes and problems."

        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            speech_analyzer._analyze_sentiment(mixed_text)
        )

        # Should detect both positive and negative words
        positive_words = [
            kw for kw in emotional_keywords if kw["sentiment"] == "positive"
        ]
        negative_words = [
            kw for kw in emotional_keywords if kw["sentiment"] == "negative"
        ]

        assert len(positive_words) > 0
        assert len(negative_words) > 0
        assert confidence > 0

    def test_sentiment_analysis_integration(
        self, speech_analyzer, sample_transcription_result, sample_scene_result
    ):
        """Test that scene analysis includes sentiment metrics."""
        result = speech_analyzer.analyze_speech(
            sample_transcription_result, sample_scene_result
        )

        # Check that sentiment metrics are included in scene results
        scene_1 = result.get_scene_metrics(1)
        assert scene_1 is not None
        assert hasattr(scene_1, "sentiment_score")
        assert hasattr(scene_1, "sentiment_label")
        assert hasattr(scene_1, "sentiment_confidence")
        assert hasattr(scene_1, "emotional_keywords")

        # Check overall metrics include sentiment data
        assert hasattr(result.overall_metrics, "overall_sentiment_score")
        assert hasattr(result.overall_metrics, "sentiment_distribution")
        assert hasattr(result.overall_metrics, "most_positive_scene")
        assert hasattr(result.overall_metrics, "most_negative_scene")
        assert hasattr(result.overall_metrics, "emotional_intensity")
        assert hasattr(result.overall_metrics, "total_emotional_keywords")

    def test_sentiment_filtering_methods(self, speech_analyzer):
        """Test scene filtering by sentiment metrics."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=150.0,
                text_content="This is an excellent presentation with great content",
                character_count=51,
                sentence_count=1,
                actual_speaking_time=9.0,
                pause_time=1.0,
                speech_ratio=0.9,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.8,  # Positive sentiment
                sentiment_label="positive",
                sentiment_confidence=0.9,
                emotional_keywords=[
                    {"word": "excellent", "sentiment": "positive", "score": 1.0},
                    {"word": "great", "sentiment": "positive", "score": 1.0},
                ],
                quality_score=0.75,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.4,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=18,
                speaking_rate_wpm=140.0,
                text_content="This section has some terrible problems and issues",
                character_count=49,
                sentence_count=1,
                actual_speaking_time=8.5,
                pause_time=1.5,
                speech_ratio=0.85,
                avg_confidence=0.92,
                low_confidence_words=1,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=-0.6,  # Negative sentiment
                sentiment_label="negative",
                sentiment_confidence=0.8,
                emotional_keywords=[
                    {"word": "terrible", "sentiment": "negative", "score": -1.0},
                    {"word": "problems", "sentiment": "negative", "score": -1.0},
                ],
                quality_score=0.75,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.4,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=3,
                start_time=20.0,
                end_time=30.0,
                duration=10.0,
                word_count=15,
                speaking_rate_wpm=130.0,
                text_content="The data shows various statistics from the report",
                character_count=47,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.90,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.05,  # Neutral sentiment
                sentiment_label="neutral",
                sentiment_confidence=0.7,
                emotional_keywords=[],
                quality_score=0.75,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.4,
                no_speech_ratio=0.1,
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=30.0,
            total_scenes=3,
            total_word_count=53,
            average_wpm=140.0,
            median_wpm=140.0,
            min_wpm=130.0,
            max_wpm=150.0,
            overall_speech_ratio=0.85,
            total_speaking_time=25.5,
            total_pause_time=4.5,
            wpm_variance=66.7,
            wpm_standard_deviation=8.2,
            total_characters=147,
            total_sentences=3,
            average_words_per_sentence=17.7,
            overall_confidence=0.92,
            low_confidence_percentage=1.9,
            total_filler_words=0,
            filler_word_rate=0.0,
            most_common_filler_words=[],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.08,  # Slightly positive overall
            sentiment_distribution={"positive": 1, "negative": 1, "neutral": 1},
            most_positive_scene=1,
            most_negative_scene=2,
            emotional_intensity=0.48,
            total_emotional_keywords=4,
            overall_quality_score=0.8,
            quality_distribution={"excellent": 0, "good": 3, "fair": 0, "poor": 0},
            average_confidence=0.92,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.8,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.5,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test sentiment filtering
        positive_scenes = result.get_scenes_by_sentiment("positive")
        assert len(positive_scenes) == 1
        assert positive_scenes[0].scene_number == 1

        negative_scenes = result.get_scenes_by_sentiment("negative")
        assert len(negative_scenes) == 1
        assert negative_scenes[0].scene_number == 2

        neutral_scenes = result.get_scenes_by_sentiment("neutral")
        assert len(neutral_scenes) == 1
        assert neutral_scenes[0].scene_number == 3

        # Test most positive/negative scenes
        most_positive = result.get_most_positive_scenes(count=2)
        assert len(most_positive) == 2
        assert most_positive[0].scene_number == 1  # Highest positive score

        most_negative = result.get_most_negative_scenes(count=2)
        assert len(most_negative) == 2
        assert most_negative[0].scene_number == 2  # Lowest (most negative) score

        # Test highly emotional scenes
        highly_emotional = result.get_highly_emotional_scenes(intensity_threshold=0.5)
        assert len(highly_emotional) == 2  # Scenes 1 and 2 have abs(sentiment) > 0.5

        # Test neutral scenes
        neutral_emotional = result.get_neutral_scenes(neutrality_threshold=0.2)
        assert len(neutral_emotional) == 1  # Only scene 3 has abs(sentiment) <= 0.2


class TestQualityAssessment:
    """Test quality assessment functionality."""

    def test_assess_transcription_quality_good_quality(self, speech_analyzer):
        """Test quality assessment with good transcription quality."""
        # Create high-quality transcription data
        transcription_result = TranscriptionResult(
            text="This is excellent quality transcription.",
            segments=[
                Segment(
                    id=0,
                    text="This is excellent quality transcription.",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-0.1,  # High quality
                    no_speech_prob=0.05,  # Low no-speech probability
                    words=[
                        WordTimestamp(word="This", start=0.0, end=0.3, confidence=0.95),
                        WordTimestamp(word="is", start=0.4, end=0.6, confidence=0.97),
                        WordTimestamp(
                            word="excellent", start=0.7, end=1.2, confidence=0.93
                        ),
                        WordTimestamp(
                            word="quality", start=1.3, end=1.8, confidence=0.96
                        ),
                        WordTimestamp(
                            word="transcription", start=1.9, end=2.8, confidence=0.94
                        ),
                    ],
                )
            ],
            language="en",
            language_probability=0.95,
            duration=5.0,
            model_used="whisper-base",
            word_count=5,
            processing_time=1.0,
        )

        scene_words = [
            {"word": "This", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "is", "start": 0.4, "end": 0.6, "confidence": 0.97},
            {"word": "excellent", "start": 0.7, "end": 1.2, "confidence": 0.93},
            {"word": "quality", "start": 1.3, "end": 1.8, "confidence": 0.96},
            {"word": "transcription", "start": 1.9, "end": 2.8, "confidence": 0.94},
        ]

        quality_result = speech_analyzer._assess_transcription_quality(
            transcription_result, 0.0, 5.0, scene_words
        )

        (
            quality_score,
            quality_category,
            confidence_issues,
            segment_scores,
            avg_logprob,
            no_speech_ratio,
            problematic_segments,
        ) = quality_result

        assert quality_score > 0.8
        assert quality_category in ["excellent", "good"]
        assert len(confidence_issues) == 0
        assert len(problematic_segments) == 0
        assert avg_logprob > -0.5
        assert no_speech_ratio < 0.2

    def test_assess_transcription_quality_poor_quality(self, speech_analyzer):
        """Test quality assessment with poor transcription quality."""
        # Create low-quality transcription data
        transcription_result = TranscriptionResult(
            text="Um... this... poor quality...",
            segments=[
                Segment(
                    id=0,
                    text="Um... this... poor quality...",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-2.5,  # Very low quality
                    no_speech_prob=0.8,  # High no-speech probability
                    words=[
                        WordTimestamp(word="Um", start=0.0, end=0.5, confidence=0.4),
                        WordTimestamp(word="this", start=1.0, end=1.3, confidence=0.3),
                        WordTimestamp(word="poor", start=2.0, end=2.4, confidence=0.2),
                        WordTimestamp(
                            word="quality", start=3.0, end=3.6, confidence=0.1
                        ),
                    ],
                )
            ],
            language="en",
            language_probability=0.6,
            duration=5.0,
            model_used="whisper-base",
            word_count=4,
            processing_time=1.0,
        )

        scene_words = [
            {"word": "Um", "start": 0.0, "end": 0.5, "confidence": 0.4},
            {"word": "this", "start": 1.0, "end": 1.3, "confidence": 0.3},
            {"word": "poor", "start": 2.0, "end": 2.4, "confidence": 0.2},
            {"word": "quality", "start": 3.0, "end": 3.6, "confidence": 0.1},
        ]

        quality_result = speech_analyzer._assess_transcription_quality(
            transcription_result, 0.0, 5.0, scene_words
        )

        (
            quality_score,
            quality_category,
            confidence_issues,
            segment_scores,
            avg_logprob,
            no_speech_ratio,
            problematic_segments,
        ) = quality_result

        assert quality_score < 0.5
        assert quality_category in ["poor", "fair"]
        assert len(confidence_issues) > 0
        assert len(problematic_segments) > 0
        assert avg_logprob < -1.0
        assert no_speech_ratio > 0.5

    def test_assess_transcription_quality_empty_scene(self, speech_analyzer):
        """Test quality assessment with empty scene data."""
        transcription_result = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            language_probability=0.95,
            duration=5.0,
            model_used="whisper-base",
            word_count=0,
            processing_time=1.0,
        )

        quality_result = speech_analyzer._assess_transcription_quality(
            transcription_result, 0.0, 5.0, []
        )

        (
            quality_score,
            quality_category,
            confidence_issues,
            segment_scores,
            avg_logprob,
            no_speech_ratio,
            problematic_segments,
        ) = quality_result

        assert quality_score == 0.0
        assert quality_category == "poor"
        assert len(confidence_issues) == 1
        assert confidence_issues[0]["issue"] == "no_words"
        assert confidence_issues[0]["severity"] == "critical"

    def test_quality_filtering_methods(self, speech_analyzer):
        """Test quality filtering methods in SpeechAnalysisResult."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,
                text_content="High quality content",
                character_count=20,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
                filler_word_count=0,
                filler_word_rate=0.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.9,
                quality_category="excellent",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.1,
                no_speech_ratio=0.05,
                problematic_segments=[],
            ),
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=15,
                speaking_rate_wpm=90.0,
                text_content="Poor quality content",
                character_count=20,
                sentence_count=1,
                actual_speaking_time=6.0,
                pause_time=4.0,
                speech_ratio=0.6,
                avg_confidence=0.45,
                low_confidence_words=8,
                filler_word_count=3,
                filler_word_rate=18.0,
                filler_words_detected=[],
                silence_segments=[],
                total_silence_time=2.0,
                silence_ratio=0.2,
                average_silence_duration=1.0,
                longest_silence_duration=1.0,
                silence_count=2,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.3,
                quality_category="poor",
                confidence_issues=[
                    {"issue": "low_confidence", "severity": "high", "value": 0.45}
                ],
                segment_confidence_scores=[],
                avg_segment_logprob=-1.5,
                no_speech_ratio=0.6,
                problematic_segments=[
                    {
                        "segment_id": 1,
                        "issues": ["low_logprob"],
                        "severity": 2,
                        "issue_type": "low_logprob",
                    }
                ],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=20.0,
            total_scenes=2,
            total_word_count=35,
            average_wpm=105.0,
            median_wpm=105.0,
            min_wpm=90.0,
            max_wpm=120.0,
            overall_speech_ratio=0.7,
            total_speaking_time=14.0,
            total_pause_time=6.0,
            wpm_variance=225.0,
            wpm_standard_deviation=15.0,
            total_characters=40,
            total_sentences=2,
            average_words_per_sentence=17.5,
            overall_confidence=0.7,
            low_confidence_percentage=22.86,
            total_filler_words=3,
            filler_word_rate=9.0,
            most_common_filler_words=[],
            total_silence_time=2.0,
            overall_silence_ratio=0.1,
            average_silence_per_scene=1.0,
            longest_silence_overall=1.0,
            total_silence_segments=2,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.6,
            quality_distribution={"excellent": 1, "good": 0, "fair": 0, "poor": 1},
            average_confidence=0.7,
            low_quality_scenes=[2],
            critical_issues=[
                {"issue": "low_confidence", "severity": "high", "scene_number": 2}
            ],
            transcription_reliability=0.6,
            recommended_actions=["Review 1 scenes with low quality transcription"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=2.0,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test quality filtering methods
        high_quality = result.get_high_quality_scenes()
        assert len(high_quality) == 1
        assert high_quality[0].scene_number == 1

        low_quality = result.get_low_quality_scenes()
        assert len(low_quality) == 1
        assert low_quality[0].scene_number == 2

        excellent_scenes = result.get_scenes_by_quality("excellent")
        assert len(excellent_scenes) == 1
        assert excellent_scenes[0].scene_number == 1

        poor_scenes = result.get_scenes_by_quality("poor")
        assert len(poor_scenes) == 1
        assert poor_scenes[0].scene_number == 2

        confidence_issues = result.get_scenes_with_confidence_issues()
        assert len(confidence_issues) == 1
        assert confidence_issues[0].scene_number == 2

        high_no_speech = result.get_scenes_with_high_no_speech_ratio()
        assert len(high_no_speech) == 1
        assert high_no_speech[0].scene_number == 2

        low_logprob = result.get_scenes_with_low_avg_logprob()
        assert len(low_logprob) == 1
        assert low_logprob[0].scene_number == 2

        # Test problematic segments summary
        summary = result.get_problematic_segments_summary()
        assert summary["total_problematic_segments"] == 1
        assert "low_logprob" in summary["issue_type_distribution"]
