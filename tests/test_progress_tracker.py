"""
Tests for progress_tracker module.

Tests the ProgressMetrics and StreamlitProgressTracker classes.
"""

import os
import sys
import time
import pytest
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from progress_tracker import ProgressMetrics, StreamlitProgressTracker


class TestProgressMetrics:
    """Tests for the ProgressMetrics dataclass."""

    def test_init_default_values(self):
        """Test ProgressMetrics initializes with default values."""
        metrics = ProgressMetrics(total_items=10)
        assert metrics.total_items == 10
        assert metrics.completed_items == 0
        assert metrics.start_time is None
        assert metrics.item_times == []

    def test_start_records_time(self):
        """Test start method records start time."""
        metrics = ProgressMetrics(total_items=5)
        metrics.start()
        assert metrics.start_time is not None
        assert isinstance(metrics.start_time, float)

    def test_record_item_completion(self):
        """Test record_item_completion updates metrics."""
        metrics = ProgressMetrics(total_items=3)
        metrics.start()

        metrics.record_item_completion(1.5)
        assert metrics.completed_items == 1
        assert len(metrics.item_times) == 1
        assert metrics.item_times[0] == 1.5

        metrics.record_item_completion(2.0)
        assert metrics.completed_items == 2
        assert len(metrics.item_times) == 2

    def test_progress_percentage_zero(self):
        """Test progress_percentage at start."""
        metrics = ProgressMetrics(total_items=10)
        assert metrics.progress_percentage == 0.0

    def test_progress_percentage_partial(self):
        """Test progress_percentage after some completion."""
        metrics = ProgressMetrics(total_items=10)
        metrics.record_item_completion(1.0)
        metrics.record_item_completion(1.0)
        assert metrics.progress_percentage == 20.0

    def test_progress_percentage_complete(self):
        """Test progress_percentage at completion."""
        metrics = ProgressMetrics(total_items=2)
        metrics.record_item_completion(1.0)
        metrics.record_item_completion(1.0)
        assert metrics.progress_percentage == 100.0

    def test_progress_percentage_zero_total(self):
        """Test progress_percentage with zero total items."""
        metrics = ProgressMetrics(total_items=0)
        assert metrics.progress_percentage == 100.0

    def test_progress_fraction(self):
        """Test progress_fraction calculation."""
        metrics = ProgressMetrics(total_items=4)
        metrics.record_item_completion(1.0)
        assert metrics.progress_fraction == 0.25

    def test_elapsed_time(self):
        """Test elapsed_time calculation."""
        metrics = ProgressMetrics(total_items=5)

        # Before start, elapsed time should be 0
        assert metrics.elapsed_time == 0.0

        metrics.start()
        time.sleep(0.1)
        elapsed = metrics.elapsed_time

        assert elapsed >= 0.1

    def test_average_item_time_empty(self):
        """Test average_item_time with no items."""
        metrics = ProgressMetrics(total_items=5)
        assert metrics.average_item_time == 0.0

    def test_average_item_time(self):
        """Test average_item_time calculation."""
        metrics = ProgressMetrics(total_items=5)
        metrics.record_item_completion(1.0)
        metrics.record_item_completion(2.0)
        metrics.record_item_completion(3.0)
        assert metrics.average_item_time == 2.0

    def test_estimated_remaining_time_empty(self):
        """Test estimated_remaining_time with no items."""
        metrics = ProgressMetrics(total_items=5)
        assert metrics.estimated_remaining_time == 0.0

    def test_estimated_remaining_time(self):
        """Test estimated_remaining_time calculation."""
        metrics = ProgressMetrics(total_items=10)
        # Complete 5 items at 2 seconds each
        for _ in range(5):
            metrics.record_item_completion(2.0)

        # Should estimate ~10 seconds remaining (5 items * 2 seconds)
        # Using EMA so might be slightly different
        remaining = metrics.estimated_remaining_time
        assert remaining > 0
        assert remaining < 20  # Should be reasonable

    def test_estimated_total_time(self):
        """Test estimated_total_time calculation."""
        metrics = ProgressMetrics(total_items=10)
        metrics.record_item_completion(2.0)
        metrics.record_item_completion(2.0)

        # 10 items * 2 seconds average = 20 seconds
        assert metrics.estimated_total_time == 20.0

    def test_format_time_seconds(self):
        """Test format_time with seconds only."""
        metrics = ProgressMetrics(total_items=1)
        assert metrics.format_time(45) == "45s"
        assert metrics.format_time(0) == "0s"

    def test_format_time_minutes(self):
        """Test format_time with minutes."""
        metrics = ProgressMetrics(total_items=1)
        assert metrics.format_time(90) == "1m 30s"
        assert metrics.format_time(120) == "2m 0s"

    def test_format_time_hours(self):
        """Test format_time with hours."""
        metrics = ProgressMetrics(total_items=1)
        assert metrics.format_time(3660) == "1h 1m"
        assert metrics.format_time(7200) == "2h 0m"

    def test_format_time_negative(self):
        """Test format_time with negative values."""
        metrics = ProgressMetrics(total_items=1)
        assert metrics.format_time(-10) == "0s"

    def test_get_status_message_start(self):
        """Test get_status_message at start."""
        metrics = ProgressMetrics(total_items=5)
        message = metrics.get_status_message()
        assert "Processing chunk 1/5" in message

    def test_get_status_message_in_progress(self):
        """Test get_status_message during progress."""
        metrics = ProgressMetrics(total_items=5)
        metrics.start()
        metrics.record_item_completion(1.0)

        message = metrics.get_status_message()
        assert "Processing chunk 2/5" in message
        assert "Elapsed:" in message
        assert "ETA:" in message


class TestStreamlitProgressTracker:
    """Tests for the StreamlitProgressTracker class."""

    @patch('progress_tracker.st')
    def test_init(self, mock_st):
        """Test StreamlitProgressTracker initialization."""
        tracker = StreamlitProgressTracker(total_items=10, description="Test")
        assert tracker.metrics.total_items == 10
        assert tracker.description == "Test"

    @patch('progress_tracker.st')
    def test_start_creates_ui_components(self, mock_st):
        """Test start method creates Streamlit UI components."""
        mock_st.progress.return_value = Mock()
        mock_st.empty.return_value = Mock()

        tracker = StreamlitProgressTracker(total_items=5, description="Test")
        tracker.start()

        mock_st.progress.assert_called_once()
        assert mock_st.empty.call_count == 2  # status_text and eta_text

    @patch('progress_tracker.st')
    def test_update_increments_progress(self, mock_st):
        """Test update method increments progress."""
        mock_progress = Mock()
        mock_status = Mock()
        mock_eta = Mock()

        mock_st.progress.return_value = mock_progress
        mock_st.empty.side_effect = [mock_status, mock_eta]

        tracker = StreamlitProgressTracker(total_items=5, description="Test")
        tracker.start()

        tracker.update(1.0)
        assert tracker.metrics.completed_items == 1

        tracker.update(1.5)
        assert tracker.metrics.completed_items == 2

    @patch('progress_tracker.st')
    def test_complete_shows_final_status(self, mock_st):
        """Test complete method shows final status."""
        mock_progress = Mock()
        mock_status = Mock()
        mock_eta = Mock()

        mock_st.progress.return_value = mock_progress
        mock_st.empty.side_effect = [mock_status, mock_eta]

        tracker = StreamlitProgressTracker(total_items=2, description="Test")
        tracker.start()
        tracker.update(1.0)
        tracker.update(1.0)
        tracker.complete()

        # Progress bar should be set to 100%
        mock_progress.progress.assert_called()

    @patch('progress_tracker.st')
    def test_clear_removes_ui_elements(self, mock_st):
        """Test clear method removes UI elements."""
        mock_progress = Mock()
        mock_status = Mock()
        mock_eta = Mock()

        mock_st.progress.return_value = mock_progress
        mock_st.empty.side_effect = [mock_status, mock_eta]

        tracker = StreamlitProgressTracker(total_items=5, description="Test")
        tracker.start()
        tracker.clear()

        mock_progress.empty.assert_called_once()
        mock_status.empty.assert_called_once()
        mock_eta.empty.assert_called_once()
