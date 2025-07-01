"""Comprehensive additional tests for speech analysis functionality.

This file contains edge case tests, error handling tests, and performance tests
to achieve complete coverage and ensure robustness of the speech analysis system.
"""

import pytest

from deep_brief.analysis.speech_analyzer import (
    OverallSpeechMetrics,
    SceneSpeechMetrics,
    SpeechAnalysisResult,
    SpeechAnalyzer,
)
from deep_brief.analysis.transcriber import (
    Segment,
    TranscriptionResult,
    WordTimestamp,
)
from deep_brief.core.scene_detector import Scene, SceneDetectionResult
from deep_brief.utils.config import AnalysisConfig, DeepBriefConfig


class TestSpeechAnalyzerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def speech_analyzer(self):
        """Create SpeechAnalyzer instance for testing."""
        config = DeepBriefConfig(
            analysis=AnalysisConfig(
                confidence_threshold=0.7,
                target_wpm_range=[140, 160],
                sentiment_analysis=True,
            )
        )
        return SpeechAnalyzer(config=config)

    def test_analyze_speech_empty_transcription(self, speech_analyzer):
        """Test analysis with empty transcription."""
        transcription_result = TranscriptionResult(
            text="",
            segments=[],
            language="en",
            language_probability=0.95,
            duration=10.0,
            model_used="whisper-base",
            word_count=0,
            processing_time=1.0,
        )

        scene_result = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=0.0,
                    end_time=10.0,
                    duration=10.0,
                    scene_number=1,
                    confidence=0.8,
                )
            ],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=10.0,
            average_scene_duration=10.0,
        )

        result = speech_analyzer.analyze_speech(transcription_result, scene_result)

        assert len(result.scene_metrics) == 1
        scene = result.scene_metrics[0]
        assert scene.word_count == 0
        assert scene.speaking_rate_wpm == 0.0
        assert scene.text_content == ""
        assert scene.sentiment_score == 0.0
        assert scene.sentiment_label == "neutral"
        assert scene.quality_score == 0.0
        assert scene.quality_category == "poor"

    def test_analyze_speech_with_extreme_sentiment(self, speech_analyzer):
        """Test analysis with very positive and negative sentiment."""
        # Create transcription with extreme sentiment
        transcription_result = TranscriptionResult(
            text="This is absolutely terrible horrible awful disappointing bad. This is excellent amazing wonderful fantastic brilliant outstanding.",
            segments=[
                Segment(
                    id=0,
                    text="This is absolutely terrible horrible awful disappointing bad.",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    words=[
                        WordTimestamp(word="This", start=0.0, end=0.3, confidence=0.95),
                        WordTimestamp(word="is", start=0.4, end=0.6, confidence=0.95),
                        WordTimestamp(
                            word="absolutely", start=0.7, end=1.2, confidence=0.95
                        ),
                        WordTimestamp(
                            word="terrible", start=1.3, end=1.8, confidence=0.95
                        ),
                        WordTimestamp(
                            word="horrible", start=1.9, end=2.4, confidence=0.95
                        ),
                        WordTimestamp(
                            word="awful", start=2.5, end=2.9, confidence=0.95
                        ),
                        WordTimestamp(
                            word="disappointing", start=3.0, end=3.8, confidence=0.95
                        ),
                        WordTimestamp(word="bad", start=3.9, end=4.2, confidence=0.95),
                    ],
                ),
                Segment(
                    id=1,
                    text="This is excellent amazing wonderful fantastic brilliant outstanding.",
                    start=10.0,
                    end=15.0,
                    avg_logprob=-0.1,
                    no_speech_prob=0.05,
                    words=[
                        WordTimestamp(
                            word="This", start=10.0, end=10.3, confidence=0.95
                        ),
                        WordTimestamp(word="is", start=10.4, end=10.6, confidence=0.95),
                        WordTimestamp(
                            word="excellent", start=10.7, end=11.2, confidence=0.95
                        ),
                        WordTimestamp(
                            word="amazing", start=11.3, end=11.8, confidence=0.95
                        ),
                        WordTimestamp(
                            word="wonderful", start=11.9, end=12.4, confidence=0.95
                        ),
                        WordTimestamp(
                            word="fantastic", start=12.5, end=13.0, confidence=0.95
                        ),
                        WordTimestamp(
                            word="brilliant", start=13.1, end=13.6, confidence=0.95
                        ),
                        WordTimestamp(
                            word="outstanding", start=13.7, end=14.5, confidence=0.95
                        ),
                    ],
                ),
            ],
            language="en",
            language_probability=0.95,
            duration=20.0,
            model_used="whisper-base",
            word_count=16,
            processing_time=2.0,
        )

        scene_result = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=0.0,
                    end_time=8.0,
                    duration=8.0,
                    scene_number=1,
                    confidence=0.8,
                ),
                Scene(
                    start_time=8.0,
                    end_time=20.0,
                    duration=12.0,
                    scene_number=2,
                    confidence=0.8,
                ),
            ],
            total_scenes=2,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=20.0,
            average_scene_duration=10.0,
        )

        result = speech_analyzer.analyze_speech(transcription_result, scene_result)

        # Test that extreme sentiments are captured
        assert len(result.scene_metrics) == 2

        # First scene should be very negative
        scene1 = result.scene_metrics[0]
        assert scene1.sentiment_score < -0.3  # Very negative
        assert scene1.sentiment_label == "negative"
        assert len(scene1.emotional_keywords) > 0

        # Second scene should be very positive
        scene2 = result.scene_metrics[1]
        assert scene2.sentiment_score > 0.3  # Very positive
        assert scene2.sentiment_label == "positive"
        assert len(scene2.emotional_keywords) > 0

        # Overall metrics should reflect extremes
        assert result.overall_metrics.most_negative_scene == 1
        assert result.overall_metrics.most_positive_scene == 2
        assert result.overall_metrics.emotional_intensity > 0.2

    def test_analyze_speech_with_zero_duration_scene(self, speech_analyzer):
        """Test analysis with zero-duration scene."""
        transcription_result = TranscriptionResult(
            text="Hello world",
            segments=[
                Segment(
                    id=0,
                    text="Hello world",
                    start=5.0,
                    end=5.0,  # Zero duration
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    words=[],
                )
            ],
            language="en",
            language_probability=0.95,
            duration=10.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=1.0,
        )

        scene_result = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=5.0,
                    end_time=5.0,  # Zero duration scene
                    duration=0.0,
                    scene_number=1,
                    confidence=0.8,
                )
            ],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=10.0,
            average_scene_duration=0.0,
        )

        result = speech_analyzer.analyze_speech(transcription_result, scene_result)

        assert len(result.scene_metrics) == 1
        scene = result.scene_metrics[0]
        assert scene.duration == 0.0
        assert scene.speech_ratio == 0.0  # Should handle division by zero

    def test_analyze_speech_segments_without_words(self, speech_analyzer):
        """Test analysis with segments that have no word-level timestamps."""
        transcription_result = TranscriptionResult(
            text="Hello world test content",
            segments=[
                Segment(
                    id=0,
                    text="Hello world test content",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    words=[],  # No word-level timestamps
                )
            ],
            language="en",
            language_probability=0.95,
            duration=10.0,
            model_used="whisper-base",
            word_count=4,
            processing_time=1.0,
        )

        scene_result = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=0.0,
                    end_time=10.0,
                    duration=10.0,
                    scene_number=1,
                    confidence=0.8,
                )
            ],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=10.0,
            average_scene_duration=10.0,
        )

        result = speech_analyzer.analyze_speech(transcription_result, scene_result)

        assert len(result.scene_metrics) == 1
        scene = result.scene_metrics[0]
        # Should still calculate basic metrics using segment-level info
        assert scene.text_content == "Hello world test content"
        assert scene.sentence_count == 1
        assert scene.character_count == 24

    def test_quality_assessment_edge_cases(self, speech_analyzer):
        """Test quality assessment with edge case scenarios."""
        # Test with segments having extreme log probabilities
        transcription_result = TranscriptionResult(
            text="Low quality transcription",
            segments=[
                Segment(
                    id=0,
                    text="Low quality transcription",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-5.0,  # Extremely low log probability
                    no_speech_prob=0.9,  # Very high no-speech probability
                    words=[
                        WordTimestamp(word="Low", start=0.0, end=0.5, confidence=0.1),
                        WordTimestamp(
                            word="quality", start=1.0, end=1.5, confidence=0.2
                        ),
                        WordTimestamp(
                            word="transcription", start=2.0, end=3.0, confidence=0.1
                        ),
                    ],
                )
            ],
            language="en",
            language_probability=0.3,  # Low language probability
            duration=5.0,
            model_used="whisper-base",
            word_count=3,
            processing_time=1.0,
        )

        scene_words = [
            {"word": "Low", "start": 0.0, "end": 0.5, "confidence": 0.1},
            {"word": "quality", "start": 1.0, "end": 1.5, "confidence": 0.2},
            {"word": "transcription", "start": 2.0, "end": 3.0, "confidence": 0.1},
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

        # Should detect very poor quality
        assert quality_score < 0.3
        assert quality_category == "poor"
        assert len(confidence_issues) > 0
        assert len(problematic_segments) > 0
        assert avg_logprob < -3.0
        assert no_speech_ratio > 0.8

    def test_silence_detection_edge_cases(self, speech_analyzer):
        """Test silence detection with edge cases."""
        # Test with words that have overlapping timestamps
        scene_words = [
            {"word": "Hello", "start": 0.0, "end": 1.0, "confidence": 0.9},
            {
                "word": "world",
                "start": 0.5,
                "end": 1.5,
                "confidence": 0.9,
            },  # Overlapping
            {"word": "test", "start": 3.0, "end": 4.0, "confidence": 0.9},
        ]

        silence_result = speech_analyzer._detect_silence_segments(
            scene_words, 0.0, 5.0, min_silence_duration=0.1
        )

        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            avg_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = silence_result

        # Should handle overlapping timestamps gracefully
        assert isinstance(silence_segments, list)
        assert total_silence_time >= 0
        assert 0 <= silence_ratio <= 1
        assert silence_count >= 0

    def test_filler_word_detection_case_sensitivity(self, speech_analyzer):
        """Test filler word detection with various cases and punctuation."""
        scene_words = [
            {"word": "Um,", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "LIKE", "start": 1.0, "end": 1.3, "confidence": 0.9},
            {"word": "you", "start": 1.4, "end": 1.6, "confidence": 0.9},
            {"word": "know?", "start": 1.7, "end": 2.0, "confidence": 0.9},
            {"word": "Uh...", "start": 2.5, "end": 2.8, "confidence": 0.9},
        ]

        filler_count, filler_rate, detected_fillers = (
            speech_analyzer._detect_filler_words(scene_words, 5.0)
        )

        # Should detect fillers regardless of case and punctuation
        assert filler_count >= 3  # um, like, uh (know may not be a filler word)
        assert len(detected_fillers) >= 3
        assert filler_rate > 0

        # Verify specific detections
        detected_words = [filler["normalized"] for filler in detected_fillers]
        assert "um" in detected_words
        assert "like" in detected_words
        assert "uh" in detected_words
        # Note: "know" may not be detected as a filler word by itself

    def test_sentiment_analysis_disabled(self, speech_analyzer):
        """Test sentiment analysis when disabled in config."""
        # Temporarily disable sentiment analysis
        speech_analyzer.config.analysis.sentiment_analysis = False

        sentiment_result = speech_analyzer._analyze_sentiment(
            "This is excellent and amazing!"
        )

        sentiment_score, sentiment_label, confidence, emotional_keywords = (
            sentiment_result
        )

        # Should return neutral when disabled
        assert sentiment_score == 0.0
        assert sentiment_label == "neutral"
        assert confidence == 0.0
        assert emotional_keywords == []

    def test_overall_metrics_with_single_scene(self, speech_analyzer):
        """Test overall metrics calculation with just one scene."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=120.0,
                text_content="Single scene content",
                character_count=20,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.95,
                low_confidence_words=0,
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
                silence_segments=[],
                total_silence_time=0.0,
                silence_ratio=0.0,
                average_silence_duration=0.0,
                longest_silence_duration=0.0,
                silence_count=0,
                sentiment_score=0.7,  # Positive
                sentiment_label="positive",
                sentiment_confidence=0.8,
                emotional_keywords=[
                    {"word": "excellent", "sentiment": "positive", "score": 1.0}
                ],
                quality_score=0.9,
                quality_category="excellent",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.1,
                no_speech_ratio=0.05,
                problematic_segments=[],
            )
        ]

        overall_metrics = speech_analyzer._calculate_overall_metrics(
            scene_metrics, 10.0
        )

        # Verify single scene metrics
        assert overall_metrics.total_scenes == 1
        assert overall_metrics.median_wpm == 120.0  # Same as average with one scene
        assert overall_metrics.min_wpm == 120.0
        assert overall_metrics.max_wpm == 120.0
        assert overall_metrics.wpm_variance == 0.0  # No variance with one scene
        assert overall_metrics.wpm_standard_deviation == 0.0

        # Should identify most positive scene
        assert overall_metrics.most_positive_scene == 1
        assert overall_metrics.most_negative_scene is None  # Not negative enough

    def test_get_total_filler_words_by_type(self, speech_analyzer):
        """Test aggregation of filler words across scenes."""
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
                filler_word_count=3,
                filler_word_rate=18.0,
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
                    {
                        "word": "um",
                        "normalized": "um",
                        "start": 8.0,
                        "end": 8.2,
                        "confidence": 0.87,
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
            SceneSpeechMetrics(
                scene_number=2,
                start_time=10.0,
                end_time=20.0,
                duration=10.0,
                word_count=15,
                speaking_rate_wpm=90.0,
                text_content="Scene two content",
                character_count=17,
                sentence_count=1,
                actual_speaking_time=6.0,
                pause_time=4.0,
                speech_ratio=0.6,
                avg_confidence=0.85,
                low_confidence_words=2,
                filler_word_count=2,
                filler_word_rate=12.0,
                filler_words_detected=[
                    {
                        "word": "like",
                        "normalized": "like",
                        "start": 12.0,
                        "end": 12.3,
                        "confidence": 0.88,
                    },
                    {
                        "word": "you",
                        "normalized": "you",
                        "start": 15.0,
                        "end": 15.2,
                        "confidence": 0.82,
                    },
                ],
                silence_segments=[],
                total_silence_time=1.0,
                silence_ratio=0.1,
                average_silence_duration=1.0,
                longest_silence_duration=1.0,
                silence_count=1,
                sentiment_score=0.0,
                sentiment_label="neutral",
                sentiment_confidence=0.5,
                emotional_keywords=[],
                quality_score=0.75,
                quality_category="good",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-0.4,
                no_speech_ratio=0.15,
                problematic_segments=[],
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
            total_characters=34,
            total_sentences=2,
            average_words_per_sentence=17.5,
            overall_confidence=0.9,
            low_confidence_percentage=5.7,
            total_filler_words=5,
            filler_word_rate=15.0,
            most_common_filler_words=[
                {"word": "um", "count": 2},
                {"word": "like", "count": 2},
                {"word": "you", "count": 1},
            ],
            total_silence_time=1.0,
            overall_silence_ratio=0.05,
            average_silence_per_scene=0.5,
            longest_silence_overall=1.0,
            total_silence_segments=1,
            overall_sentiment_score=0.0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 2},
            most_positive_scene=None,
            most_negative_scene=None,
            emotional_intensity=0.0,
            total_emotional_keywords=0,
            overall_quality_score=0.8,
            quality_distribution={"excellent": 0, "good": 2, "fair": 0, "poor": 0},
            average_confidence=0.9,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.8,
            recommended_actions=["Transcription quality is good"],
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

        # Test filler word aggregation
        filler_counts = result.get_total_filler_words_by_type()

        assert filler_counts["um"] == 2
        assert filler_counts["like"] == 2
        assert filler_counts["you"] == 1
        assert len(filler_counts) == 3

    def test_scene_filtering_boundary_conditions(self, speech_analyzer):
        """Test scene filtering methods with boundary values."""
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=20,
                speaking_rate_wpm=140.0,  # Exactly at lower boundary
                text_content="Boundary test content",
                character_count=21,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.7,  # Exactly at threshold
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
                sentiment_score=0.1,  # Just above neutral threshold
                sentiment_label="positive",
                sentiment_confidence=0.6,
                emotional_keywords=[],
                quality_score=0.6,  # Exactly at boundary
                quality_category="fair",
                confidence_issues=[],
                segment_confidence_scores=[],
                avg_segment_logprob=-1.0,  # Exactly at threshold
                no_speech_ratio=0.3,  # Exactly at threshold
                problematic_segments=[],
            ),
        ]

        overall_metrics = OverallSpeechMetrics(
            total_duration=10.0,
            total_scenes=1,
            total_word_count=20,
            average_wpm=140.0,
            median_wpm=140.0,
            min_wpm=140.0,
            max_wpm=140.0,
            overall_speech_ratio=0.8,
            total_speaking_time=8.0,
            total_pause_time=2.0,
            wpm_variance=0.0,
            wpm_standard_deviation=0.0,
            total_characters=21,
            total_sentences=1,
            average_words_per_sentence=20.0,
            overall_confidence=0.7,
            low_confidence_percentage=0.0,
            total_filler_words=0,
            filler_word_rate=0.0,
            most_common_filler_words=[],
            total_silence_time=0.0,
            overall_silence_ratio=0.0,
            average_silence_per_scene=0.0,
            longest_silence_overall=0.0,
            total_silence_segments=0,
            overall_sentiment_score=0.1,
            sentiment_distribution={"positive": 1, "negative": 0, "neutral": 0},
            most_positive_scene=1,
            most_negative_scene=None,
            emotional_intensity=0.1,
            total_emotional_keywords=0,
            overall_quality_score=0.6,
            quality_distribution={"excellent": 0, "good": 0, "fair": 1, "poor": 0},
            average_confidence=0.7,
            low_quality_scenes=[1],
            critical_issues=[],
            transcription_reliability=0.6,
            recommended_actions=["Review 1 scenes with low quality transcription"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.0,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test boundary conditions for various filters
        optimal_scenes = result.get_optimal_scenes()
        assert len(optimal_scenes) == 1  # 140 is exactly at the lower boundary

        low_quality_scenes = result.get_low_quality_scenes(max_quality_score=0.6)
        assert len(low_quality_scenes) == 1  # 0.6 equals threshold

        high_no_speech = result.get_scenes_with_high_no_speech_ratio(threshold=0.29)
        assert len(high_no_speech) == 1  # 0.3 > 0.29 threshold

        low_logprob = result.get_scenes_with_low_avg_logprob(threshold=-0.9)
        assert len(low_logprob) == 1  # -1.0 < -0.9 threshold


class TestSpeechAnalyzerPerformance:
    """Test performance and scalability."""

    @pytest.fixture
    def speech_analyzer(self):
        """Create SpeechAnalyzer instance for testing."""
        return SpeechAnalyzer()

    def test_large_transcription_performance(self, speech_analyzer):
        """Test analysis with large number of scenes and words."""
        # Create large transcription with many scenes
        segments = []
        scenes = []
        word_id = 0

        for i in range(50):  # 50 scenes
            start_time = i * 10.0
            end_time = (i + 1) * 10.0

            # Create segment with many words
            words = []
            text_parts = []
            for j in range(20):  # 20 words per scene
                word_start = start_time + j * 0.5
                word_end = word_start + 0.4
                word_text = f"word{word_id}"
                words.append(
                    WordTimestamp(
                        word=word_text,
                        start=word_start,
                        end=word_end,
                        confidence=0.9,
                    )
                )
                text_parts.append(word_text)
                word_id += 1

            segment_text = " ".join(text_parts)
            segments.append(
                Segment(
                    id=i,
                    text=segment_text,
                    start=start_time,
                    end=end_time,
                    avg_logprob=-0.3,
                    no_speech_prob=0.1,
                    words=words,
                )
            )

            scenes.append(
                Scene(
                    start_time=start_time,
                    end_time=end_time,
                    duration=10.0,
                    scene_number=i + 1,
                    confidence=0.8,
                )
            )

        transcription_result = TranscriptionResult(
            text=" ".join(segment.text for segment in segments),
            segments=segments,
            language="en",
            language_probability=0.95,
            duration=500.0,
            model_used="whisper-base",
            word_count=1000,
            processing_time=10.0,
        )

        scene_result = SceneDetectionResult(
            scenes=scenes,
            total_scenes=50,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=500.0,
            average_scene_duration=10.0,
        )

        # Test that analysis completes in reasonable time
        import time

        start_time = time.time()
        result = speech_analyzer.analyze_speech(transcription_result, scene_result)
        end_time = time.time()

        # Should complete within reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0

        # Verify results are complete
        assert len(result.scene_metrics) == 50
        assert result.overall_metrics.total_scenes == 50
        assert result.overall_metrics.total_word_count == 1000

    def test_empty_scenes_handling(self, speech_analyzer):
        """Test handling of empty scenes in various calculations."""
        # Create minimal valid scene data
        scene_metrics = []

        overall_metrics = speech_analyzer._calculate_overall_metrics(
            scene_metrics, 10.0
        )

        # Should handle empty scene list gracefully
        assert overall_metrics.total_scenes == 0
        assert overall_metrics.total_word_count == 0
        assert overall_metrics.average_wpm == 0.0
        assert overall_metrics.overall_speech_ratio == 0.0
        assert overall_metrics.overall_sentiment_score == 0.0
        assert overall_metrics.emotional_intensity == 0.0


class TestSpeechAnalyzerErrorHandling:
    """Test error handling and edge case robustness."""

    @pytest.fixture
    def speech_analyzer(self):
        """Create SpeechAnalyzer instance for testing."""
        return SpeechAnalyzer()

    def test_invalid_confidence_threshold(self, speech_analyzer):
        """Test analysis with invalid confidence threshold values."""
        transcription_result = TranscriptionResult(
            text="Test content",
            segments=[
                Segment(
                    id=0,
                    text="Test content",
                    start=0.0,
                    end=5.0,
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                    words=[
                        WordTimestamp(word="Test", start=0.0, end=1.0, confidence=0.9),
                        WordTimestamp(
                            word="content", start=1.5, end=2.5, confidence=0.8
                        ),
                    ],
                )
            ],
            language="en",
            language_probability=0.95,
            duration=5.0,
            model_used="whisper-base",
            word_count=2,
            processing_time=1.0,
        )

        scene_result = SceneDetectionResult(
            scenes=[
                Scene(
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0,
                    scene_number=1,
                    confidence=0.8,
                )
            ],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=5.0,
            average_scene_duration=5.0,
        )

        # Test with extreme confidence thresholds
        result1 = speech_analyzer.analyze_speech(
            transcription_result, scene_result, confidence_threshold=0.0
        )
        result2 = speech_analyzer.analyze_speech(
            transcription_result, scene_result, confidence_threshold=1.0
        )

        # Both should complete without errors
        assert len(result1.scene_metrics) == 1
        assert len(result2.scene_metrics) == 1

        # With threshold 1.0, no words should pass confidence filter
        assert result2.scene_metrics[0].word_count == 0

    def test_to_dict_serialization(self, speech_analyzer):
        """Test dictionary serialization of results."""
        # Create minimal result
        scene_metrics = [
            SceneSpeechMetrics(
                scene_number=1,
                start_time=0.0,
                end_time=10.0,
                duration=10.0,
                word_count=5,
                speaking_rate_wpm=30.0,
                text_content="Test content",
                character_count=12,
                sentence_count=1,
                actual_speaking_time=8.0,
                pause_time=2.0,
                speech_ratio=0.8,
                avg_confidence=0.9,
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
            total_word_count=5,
            average_wpm=30.0,
            median_wpm=30.0,
            min_wpm=30.0,
            max_wpm=30.0,
            overall_speech_ratio=0.8,
            total_speaking_time=8.0,
            total_pause_time=2.0,
            wpm_variance=0.0,
            wpm_standard_deviation=0.0,
            total_characters=12,
            total_sentences=1,
            average_words_per_sentence=5.0,
            overall_confidence=0.9,
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
            average_confidence=0.9,
            low_quality_scenes=[],
            critical_issues=[],
            transcription_reliability=0.85,
            recommended_actions=["Transcription quality is good"],
        )

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=1.0,
            language="en",
            model_used="whisper-base",
            confidence_threshold=0.7,
            target_wpm_range=[140, 160],
        )

        # Test serialization
        result_dict = result.to_dict()

        # Verify structure
        assert "scene_metrics" in result_dict
        assert "overall_metrics" in result_dict
        assert "analysis_summary" in result_dict
        assert "processing_time" in result_dict
        assert "language" in result_dict
        assert "model_used" in result_dict

        # Verify analysis summary includes all expected fields
        summary = result_dict["analysis_summary"]
        assert "total_scenes" in summary
        assert "overall_quality_score" in summary
        assert "transcription_reliability" in summary
        assert "problematic_segments_summary" in summary
