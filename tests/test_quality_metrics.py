"""
Tests for quality_metrics module.

Tests the QualityAnalyzer class and QualityMetrics dataclass.
"""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality_metrics import QualityAnalyzer, QualityMetrics, get_quality_score_summary


class TestQualityAnalyzer:
    """Tests for the QualityAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a QualityAnalyzer instance for testing."""
        return QualityAnalyzer()

    def test_count_words_empty(self, analyzer):
        """Test count_words with empty string."""
        assert analyzer.count_words("") == 0
        assert analyzer.count_words(None) == 0

    def test_count_words_simple(self, analyzer):
        """Test count_words with simple text."""
        assert analyzer.count_words("Hello world") == 2
        assert analyzer.count_words("One two three four five") == 5

    def test_count_words_with_punctuation(self, analyzer):
        """Test count_words handles punctuation correctly."""
        assert analyzer.count_words("Hello, world! How are you?") == 5

    def test_count_sentences_empty(self, analyzer):
        """Test count_sentences with empty string."""
        assert analyzer.count_sentences("") == 0
        assert analyzer.count_sentences(None) == 0

    def test_count_sentences_simple(self, analyzer):
        """Test count_sentences with simple text."""
        assert analyzer.count_sentences("Hello world.") == 1
        assert analyzer.count_sentences("First sentence. Second sentence.") == 2

    def test_count_sentences_various_punctuation(self, analyzer):
        """Test count_sentences with various sentence endings."""
        text = "Is this a question? Yes it is! And this is a statement."
        assert analyzer.count_sentences(text) == 3

    def test_extract_key_terms_empty(self, analyzer):
        """Test extract_key_terms with empty string."""
        assert analyzer.extract_key_terms("") == []
        assert analyzer.extract_key_terms(None) == []

    def test_extract_key_terms_simple(self, analyzer):
        """Test extract_key_terms extracts meaningful terms."""
        text = "Machine learning algorithms process data efficiently. Machine learning is powerful."
        terms = analyzer.extract_key_terms(text, top_n=5)
        # 'machine' and 'learning' should appear due to frequency
        assert len(terms) > 0
        assert 'machine' in terms or 'learning' in terms

    def test_extract_key_terms_filters_stop_words(self, analyzer):
        """Test extract_key_terms filters common stop words."""
        text = "This is that which were been would should their there"
        terms = analyzer.extract_key_terms(text)
        # Stop words should be filtered
        assert 'this' not in terms
        assert 'that' not in terms
        assert 'which' not in terms

    def test_calculate_compression_ratio_zero_original(self, analyzer):
        """Test compression ratio with zero original length."""
        assert analyzer.calculate_compression_ratio(0, 100) == 0.0

    def test_calculate_compression_ratio_normal(self, analyzer):
        """Test compression ratio calculation."""
        # 200 chars summary from 1000 chars original = 20%
        ratio = analyzer.calculate_compression_ratio(1000, 200)
        assert ratio == 20.0

    def test_calculate_compression_ratio_same_length(self, analyzer):
        """Test compression ratio when lengths are equal."""
        ratio = analyzer.calculate_compression_ratio(500, 500)
        assert ratio == 100.0

    def test_calculate_key_term_retention(self, analyzer):
        """Test key term retention calculation."""
        original = "Machine learning algorithms process large datasets efficiently."
        summary = "Machine learning processes data efficiently."
        retention = analyzer.calculate_key_term_retention(original, summary)
        # Should have some retention since 'machine', 'learning', 'efficiently' appear in both
        assert 0 <= retention <= 100

    def test_calculate_readability_score_empty(self, analyzer):
        """Test readability score with empty text."""
        assert analyzer.calculate_readability_score("") == 0.0

    def test_calculate_readability_score_simple(self, analyzer):
        """Test readability score with simple text."""
        # Simple, clear text should have good readability
        text = "The cat sat on the mat. It was a sunny day. The birds sang."
        score = analyzer.calculate_readability_score(text)
        assert 0 <= score <= 100

    def test_analyze_returns_metrics(self, analyzer):
        """Test analyze returns QualityMetrics object."""
        original = "This is the original document with multiple sentences. It contains several paragraphs of text."
        summary = "Brief summary of the document."

        metrics = analyzer.analyze(original, summary)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.original_length > 0
        assert metrics.summary_length > 0
        assert metrics.compression_ratio > 0

    def test_analyze_compression_ratio_correct(self, analyzer):
        """Test analyze calculates correct compression ratio."""
        original = "A" * 1000
        summary = "B" * 100

        metrics = analyzer.analyze(original, summary)

        assert metrics.original_length == 1000
        assert metrics.summary_length == 100
        assert metrics.compression_ratio == 10.0


class TestQualityMetrics:
    """Tests for the QualityMetrics dataclass."""

    def test_to_dict(self):
        """Test QualityMetrics.to_dict returns correct dictionary."""
        metrics = QualityMetrics(
            original_length=1000,
            summary_length=200,
            original_word_count=150,
            summary_word_count=30,
            compression_ratio=20.0,
            sentence_count=5,
            avg_sentence_length=6.0,
            key_terms_retained=75.5,
            readability_score=80.3
        )

        result = metrics.to_dict()

        assert result['original_length'] == 1000
        assert result['summary_length'] == 200
        assert result['compression_ratio'] == 20.0
        assert result['key_terms_retained'] == 75.5
        assert result['readability_score'] == 80.3


class TestGetQualityScoreSummary:
    """Tests for get_quality_score_summary function."""

    def test_get_quality_score_summary_good_metrics(self):
        """Test quality score summary with good metrics."""
        metrics = QualityMetrics(
            original_length=1000,
            summary_length=150,
            original_word_count=150,
            summary_word_count=25,
            compression_ratio=15.0,  # Good: between 5-30%
            sentence_count=3,
            avg_sentence_length=8.0,
            key_terms_retained=80.0,  # Good retention
            readability_score=85.0    # Good readability
        )

        summary = get_quality_score_summary(metrics)

        assert "Quality Score:" in summary
        # Score should be high with good metrics

    def test_get_quality_score_summary_poor_metrics(self):
        """Test quality score summary with poor metrics."""
        metrics = QualityMetrics(
            original_length=1000,
            summary_length=900,
            original_word_count=150,
            summary_word_count=135,
            compression_ratio=90.0,  # Poor: too high
            sentence_count=20,
            avg_sentence_length=7.0,
            key_terms_retained=20.0,  # Poor retention
            readability_score=30.0    # Poor readability
        )

        summary = get_quality_score_summary(metrics)

        assert "Quality Score:" in summary
