"""
Progress tracking module for the Document Summarizer application.

Provides utilities for tracking progress, estimating time remaining,
and displaying real-time status updates in Streamlit.
"""

import time
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """
    Container for progress tracking metrics.

    Attributes:
        total_items: Total number of items to process
        completed_items: Number of items completed
        start_time: Timestamp when processing started
        item_times: List of time taken for each completed item
    """
    total_items: int
    completed_items: int = 0
    start_time: Optional[float] = None
    item_times: list = field(default_factory=list)

    def start(self) -> None:
        """Record the start time of processing."""
        self.start_time = time.time()
        logger.debug(f"Progress tracking started for {self.total_items} items")

    def record_item_completion(self, item_time: float) -> None:
        """
        Record completion of a single item.

        Args:
            item_time: Time taken to process the item in seconds
        """
        self.item_times.append(item_time)
        self.completed_items += 1
        logger.debug(
            f"Item {self.completed_items}/{self.total_items} completed in {item_time:.2f}s"
        )

    @property
    def progress_percentage(self) -> float:
        """Calculate current progress as a percentage (0-100)."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    @property
    def progress_fraction(self) -> float:
        """Calculate current progress as a fraction (0-1)."""
        if self.total_items == 0:
            return 1.0
        return self.completed_items / self.total_items

    @property
    def elapsed_time(self) -> float:
        """Calculate total elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def average_item_time(self) -> float:
        """Calculate average time per item in seconds."""
        if not self.item_times:
            return 0.0
        return sum(self.item_times) / len(self.item_times)

    @property
    def estimated_remaining_time(self) -> float:
        """
        Estimate remaining time in seconds.

        Uses exponential moving average for more accurate estimation
        when recent items are more representative of future items.
        """
        if not self.item_times:
            return 0.0

        remaining_items = self.total_items - self.completed_items
        if remaining_items <= 0:
            return 0.0

        # Use exponential moving average for recent items
        if len(self.item_times) >= 3:
            # Weight recent items more heavily
            recent_times = self.item_times[-3:]
            weights = [0.2, 0.3, 0.5]  # More weight to most recent
            avg_time = sum(t * w for t, w in zip(recent_times, weights))
        else:
            avg_time = self.average_item_time

        return avg_time * remaining_items

    @property
    def estimated_total_time(self) -> float:
        """Estimate total processing time in seconds."""
        if not self.item_times:
            return 0.0
        return self.average_item_time * self.total_items

    def format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "2m 30s" or "45s")
        """
        if seconds < 0:
            return "0s"

        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_status_message(self) -> str:
        """
        Generate a status message for display.

        Returns:
            Human-readable status message with progress and ETA
        """
        if self.completed_items == 0:
            return f"Processing chunk 1/{self.total_items}..."

        eta = self.format_time(self.estimated_remaining_time)
        elapsed = self.format_time(self.elapsed_time)

        return (
            f"Processing chunk {self.completed_items + 1}/{self.total_items} | "
            f"Elapsed: {elapsed} | ETA: {eta}"
        )


class StreamlitProgressTracker:
    """
    Progress tracker with Streamlit UI integration.

    Provides progress bar, status text, and ETA display for long-running
    operations in Streamlit applications.
    """

    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize the progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description text for the progress display
        """
        self.metrics = ProgressMetrics(total_items=total_items)
        self.description = description

        # Streamlit UI components (initialized when tracking starts)
        self._progress_bar = None
        self._status_text = None
        self._eta_text = None

        logger.info(
            f"Progress tracker initialized: {description} ({total_items} items)"
        )

    def start(self) -> None:
        """Start progress tracking and initialize UI components."""
        self.metrics.start()

        # Create Streamlit UI components
        self._progress_bar = st.progress(0)
        self._status_text = st.empty()
        self._eta_text = st.empty()

        # Show initial status
        self._update_display()
        logger.info(f"Progress tracking started: {self.description}")

    def update(self, item_time: float) -> None:
        """
        Update progress after completing an item.

        Args:
            item_time: Time taken to process the item in seconds
        """
        self.metrics.record_item_completion(item_time)
        self._update_display()

    def _update_display(self) -> None:
        """Update Streamlit UI with current progress."""
        if self._progress_bar is None:
            return

        # Update progress bar
        self._progress_bar.progress(
            self.metrics.progress_fraction,
            text=f"{self.description}: {self.metrics.progress_percentage:.0f}%"
        )

        # Update status text
        self._status_text.text(self.metrics.get_status_message())

        # Update ETA display
        if self.metrics.completed_items > 0:
            eta = self.metrics.format_time(self.metrics.estimated_remaining_time)
            avg_time = self.metrics.format_time(self.metrics.average_item_time)
            self._eta_text.caption(
                f"⏱️ Avg per chunk: {avg_time} | ⏳ Estimated remaining: {eta}"
            )

    def complete(self) -> None:
        """Mark progress as complete and show final status."""
        if self._progress_bar is not None:
            self._progress_bar.progress(1.0, text=f"{self.description}: Complete!")

        if self._status_text is not None:
            total_time = self.metrics.format_time(self.metrics.elapsed_time)
            self._status_text.text(
                f"✅ Completed {self.metrics.total_items} chunks in {total_time}"
            )

        if self._eta_text is not None:
            avg_time = self.metrics.format_time(self.metrics.average_item_time)
            self._eta_text.caption(f"📊 Average time per chunk: {avg_time}")

        logger.info(
            f"Progress tracking complete: {self.description} "
            f"({self.metrics.total_items} items in {self.metrics.elapsed_time:.2f}s)"
        )

    def clear(self) -> None:
        """Clear all progress UI elements."""
        if self._progress_bar is not None:
            self._progress_bar.empty()
        if self._status_text is not None:
            self._status_text.empty()
        if self._eta_text is not None:
            self._eta_text.empty()


def track_progress(
    items: list,
    process_func: Callable[[Any], Any],
    description: str = "Processing"
) -> list:
    """
    Process items with progress tracking.

    A convenience function that wraps item processing with progress tracking.

    Args:
        items: List of items to process
        process_func: Function to call for each item
        description: Description for the progress display

    Returns:
        List of results from processing each item

    Example:
        >>> results = track_progress(
        ...     chunks,
        ...     lambda chunk: llm.invoke(chunk),
        ...     "Summarizing chunks"
        ... )
    """
    tracker = StreamlitProgressTracker(len(items), description)
    tracker.start()

    results = []
    for item in items:
        item_start = time.time()
        result = process_func(item)
        item_time = time.time() - item_start

        results.append(result)
        tracker.update(item_time)

    tracker.complete()
    return results
