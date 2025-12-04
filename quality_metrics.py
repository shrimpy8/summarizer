"""
Summary quality metrics module for the Document Summarizer application.

Provides utilities for calculating and displaying summary quality scores,
including compression ratio, readability metrics, and content analysis.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    Container for summary quality metrics.

    Attributes:
        original_length: Character count of original document
        summary_length: Character count of summary
        original_word_count: Word count of original document
        summary_word_count: Word count of summary
        compression_ratio: Ratio of summary to original length
        sentence_count: Number of sentences in summary
        avg_sentence_length: Average words per sentence
        key_terms_retained: Estimated percentage of key terms retained
        readability_score: Simple readability score (0-100)
    """
    original_length: int
    summary_length: int
    original_word_count: int
    summary_word_count: int
    compression_ratio: float
    sentence_count: int
    avg_sentence_length: float
    key_terms_retained: float
    readability_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/display."""
        return {
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "original_word_count": self.original_word_count,
            "summary_word_count": self.summary_word_count,
            "compression_ratio": round(self.compression_ratio, 2),
            "sentence_count": self.sentence_count,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "key_terms_retained": round(self.key_terms_retained, 1),
            "readability_score": round(self.readability_score, 1)
        }


class QualityAnalyzer:
    """
    Analyzes summary quality and provides scoring metrics.

    This class computes various metrics to assess the quality of a generated
    summary compared to the original document.
    """

    def __init__(self):
        """Initialize the quality analyzer."""
        logger.debug("QualityAnalyzer initialized")

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words in text.

        Args:
            text: Input text

        Returns:
            Number of words
        """
        if not text:
            return 0
        words = text.split()
        return len(words)

    @staticmethod
    def count_sentences(text: str) -> int:
        """
        Count sentences in text.

        Args:
            text: Input text

        Returns:
            Number of sentences
        """
        if not text:
            return 0
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)

    @staticmethod
    def extract_key_terms(text: str, top_n: int = 20) -> List[str]:
        """
        Extract key terms from text based on frequency.

        Args:
            text: Input text
            top_n: Number of top terms to return

        Returns:
            List of key terms
        """
        if not text:
            return []

        # Simple word frequency approach
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Common stop words to filter
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'were', 'been', 'will',
            'would', 'could', 'should', 'their', 'there', 'which', 'about',
            'into', 'more', 'than', 'also', 'some', 'such', 'when', 'what',
            'they', 'your', 'other', 'very', 'just', 'each', 'only', 'most',
            'both', 'then', 'over', 'after', 'before', 'while', 'where'
        }

        # Filter and count
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:top_n]]

    def calculate_compression_ratio(
        self, original_length: int, summary_length: int
    ) -> float:
        """
        Calculate compression ratio.

        Args:
            original_length: Length of original text
            summary_length: Length of summary

        Returns:
            Compression ratio as percentage (0-100)
        """
        if original_length == 0:
            return 0.0
        return (summary_length / original_length) * 100

    def calculate_key_term_retention(
        self, original_text: str, summary_text: str
    ) -> float:
        """
        Calculate percentage of key terms from original retained in summary.

        Args:
            original_text: Original document text
            summary_text: Summary text

        Returns:
            Percentage of key terms retained (0-100)
        """
        original_terms = set(self.extract_key_terms(original_text))
        if not original_terms:
            return 100.0

        summary_terms = set(self.extract_key_terms(summary_text, top_n=50))

        # Count how many original key terms appear in summary
        retained = original_terms.intersection(summary_terms)
        retention_rate = (len(retained) / len(original_terms)) * 100

        return retention_rate

    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate a simple readability score (0-100).

        Based on average sentence length and word complexity.
        Higher scores indicate more readable text.

        Args:
            text: Input text

        Returns:
            Readability score (0-100)
        """
        if not text:
            return 0.0

        words = text.split()
        sentences = self.count_sentences(text)

        if sentences == 0 or len(words) == 0:
            return 0.0

        # Average sentence length
        avg_sentence_len = len(words) / sentences

        # Average word length (proxy for complexity)
        avg_word_len = sum(len(word) for word in words) / len(words)

        # Score calculation:
        # - Ideal sentence length: 15-20 words
        # - Ideal word length: 4-6 characters
        # Higher deviation from ideal = lower score

        sentence_score = 100 - abs(avg_sentence_len - 17.5) * 3
        word_score = 100 - abs(avg_word_len - 5) * 10

        # Combine scores
        readability = (sentence_score + word_score) / 2

        # Clamp to 0-100
        return max(0.0, min(100.0, readability))

    def analyze(self, original_text: str, summary_text: str) -> QualityMetrics:
        """
        Analyze summary quality and return metrics.

        Args:
            original_text: Original document text
            summary_text: Generated summary text

        Returns:
            QualityMetrics object with all computed metrics
        """
        logger.info("Analyzing summary quality metrics")

        original_length = len(original_text)
        summary_length = len(summary_text)
        original_word_count = self.count_words(original_text)
        summary_word_count = self.count_words(summary_text)

        compression_ratio = self.calculate_compression_ratio(
            original_length, summary_length
        )

        sentence_count = self.count_sentences(summary_text)
        avg_sentence_length = (
            summary_word_count / sentence_count if sentence_count > 0 else 0
        )

        key_terms_retained = self.calculate_key_term_retention(
            original_text, summary_text
        )

        readability_score = self.calculate_readability_score(summary_text)

        metrics = QualityMetrics(
            original_length=original_length,
            summary_length=summary_length,
            original_word_count=original_word_count,
            summary_word_count=summary_word_count,
            compression_ratio=compression_ratio,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            key_terms_retained=key_terms_retained,
            readability_score=readability_score
        )

        # Log metrics for monitoring
        logger.info(f"Quality metrics: {metrics.to_dict()}")

        return metrics


def display_quality_metrics(metrics: QualityMetrics) -> None:
    """
    Display quality metrics in Streamlit UI.

    Args:
        metrics: QualityMetrics object to display
    """
    st.markdown("---")
    st.markdown("### 📊 Summary Quality Metrics")

    # Create three columns for metrics display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Compression Ratio",
            value=f"{metrics.compression_ratio:.1f}%",
            help="Summary length as percentage of original"
        )
        st.metric(
            label="Original Words",
            value=f"{metrics.original_word_count:,}",
            help="Word count of original document"
        )

    with col2:
        st.metric(
            label="Summary Words",
            value=f"{metrics.summary_word_count:,}",
            help="Word count of generated summary"
        )
        st.metric(
            label="Sentences",
            value=metrics.sentence_count,
            help="Number of sentences in summary"
        )

    with col3:
        st.metric(
            label="Key Terms Retained",
            value=f"{metrics.key_terms_retained:.0f}%",
            help="Percentage of important terms preserved"
        )
        st.metric(
            label="Readability Score",
            value=f"{metrics.readability_score:.0f}/100",
            help="Higher scores indicate easier reading"
        )

    # Quality assessment
    st.markdown("#### Quality Assessment")

    quality_indicators = []

    # Compression ratio assessment
    if 5 <= metrics.compression_ratio <= 30:
        quality_indicators.append("✅ Good compression ratio")
    elif metrics.compression_ratio < 5:
        quality_indicators.append("⚠️ Summary may be too short")
    else:
        quality_indicators.append("⚠️ Summary could be more concise")

    # Key terms retention assessment
    if metrics.key_terms_retained >= 60:
        quality_indicators.append("✅ Good key term retention")
    elif metrics.key_terms_retained >= 40:
        quality_indicators.append("⚠️ Moderate key term retention")
    else:
        quality_indicators.append("⚠️ Low key term retention")

    # Readability assessment
    if metrics.readability_score >= 70:
        quality_indicators.append("✅ Good readability")
    elif metrics.readability_score >= 50:
        quality_indicators.append("⚠️ Moderate readability")
    else:
        quality_indicators.append("⚠️ Complex sentence structure")

    # Display indicators
    for indicator in quality_indicators:
        st.caption(indicator)

    logger.debug(f"Quality metrics displayed: {metrics.to_dict()}")


def get_quality_score_summary(metrics: QualityMetrics) -> str:
    """
    Generate a brief quality score summary.

    Args:
        metrics: QualityMetrics object

    Returns:
        Brief summary string
    """
    # Calculate overall score (weighted average)
    compression_score = (
        100 if 5 <= metrics.compression_ratio <= 30
        else 50 if metrics.compression_ratio < 50
        else 30
    )
    overall_score = (
        compression_score * 0.3 +
        metrics.key_terms_retained * 0.4 +
        metrics.readability_score * 0.3
    )

    return f"Quality Score: {overall_score:.0f}/100"
