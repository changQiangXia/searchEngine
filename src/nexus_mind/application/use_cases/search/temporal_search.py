"""Temporal Search - Search with time-based filtering and analysis.

Enables searching within specific time ranges, finding trends over time,
and temporal navigation of the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class TimeRange:
    """A time range for filtering.
    
    Attributes:
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
    """
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within range."""
        if self.start and timestamp < self.start:
            return False
        if self.end and timestamp > self.end:
            return False
        return True
    
    @classmethod
    def from_string(
        cls,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fmt: str = "%Y-%m-%d",
    ) -> TimeRange:
        """Create TimeRange from string dates.
        
        Args:
            start: Start date string
            end: End date string
            fmt: Date format string
            
        Returns:
            TimeRange object
        """
        start_dt = datetime.strptime(start, fmt) if start else None
        end_dt = datetime.strptime(end, fmt) if end else None
        return cls(start=start_dt, end=end_dt)
    
    @classmethod
    def last_n_days(cls, n: int) -> TimeRange:
        """Create TimeRange for last n days."""
        end = datetime.now()
        start = end - timedelta(days=n)
        return cls(start=start, end=end)


@dataclass
class TemporalBucket:
    """A bucket of results within a time period.
    
    Attributes:
        start: Bucket start time
        end: Bucket end time
        count: Number of items in bucket
        items: List of items in bucket
        trend: Trend indicator ("up", "down", "stable")
    """
    start: datetime
    end: datetime
    count: int
    items: List[Dict[str, Any]]
    trend: str = "stable"


class TemporalSearch:
    """Search with temporal awareness.
    
    This class extends semantic search with time-based filtering
    and analysis capabilities.
    
    Example:
        >>> ts = TemporalSearch(engine)
        >>> # Search last 7 days
        >>> results = ts.search("sunset", TimeRange.last_n_days(7))
        >>> 
        >>> # Search specific date range
        >>> range = TimeRange.from_string("2024-01-01", "2024-01-31")
        >>> results = ts.search("beach", range)
    
    Attributes:
        engine: NexusEngine instance
        date_field: Metadata field containing timestamp
    """
    
    def __init__(
        self,
        engine: Any,
        date_field: str = "timestamp",
        date_format: Optional[str] = None,
    ):
        """Initialize temporal search.
        
        Args:
            engine: NexusEngine instance
            date_field: Field name for timestamps in metadata
            date_format: Expected date format (None for auto-detect)
        """
        self.engine = engine
        self.date_field = date_field
        self.date_format = date_format
    
    def search(
        self,
        query: Union[str, Any],
        time_range: Optional[TimeRange] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search within time range.
        
        Args:
            query: Search query (text or image)
            time_range: Optional time range filter
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        # Get initial results (may need more to filter)
        initial_k = top_k * 3 if time_range else top_k
        results = self.engine.search(query, top_k=initial_k)
        
        if not time_range:
            return results[:top_k]
        
        # Filter by time
        filtered = []
        for r in results:
            timestamp = self._extract_timestamp(r.get("metadata", {}))
            if timestamp and time_range.contains(timestamp):
                filtered.append(r)
            if len(filtered) >= top_k:
                break
        
        return filtered
    
    def _extract_timestamp(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from metadata."""
        timestamp = metadata.get(self.date_field)
        if timestamp is None:
            # Try common alternatives
            for field in ["date", "created", "modified", "time"]:
                timestamp = metadata.get(field)
                if timestamp:
                    break
        
        if timestamp is None:
            return None
        
        # Parse based on type
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            return self._parse_date(timestamp)
        elif isinstance(timestamp, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(timestamp)
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple formats."""
        formats = [
            self.date_format,
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y%m%d_%H%M%S",
            "%Y%m%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%Y:%m:%d %H:%M:%S",  # EXIF format
        ]
        
        for fmt in formats:
            if fmt is None:
                continue
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def temporal_distribution(
        self,
        query: Optional[str] = None,
        bucket_size: str = "day",
    ) -> List[TemporalBucket]:
        """Get distribution of results over time.
        
        Args:
            query: Optional query to filter by
            bucket_size: "hour", "day", "week", "month", or "year"
            
        Returns:
            List of temporal buckets
        """
        # Get all results (or subset if query provided)
        if query:
            results = self.engine.search(query, top_k=1000)
        else:
            # Get all indexed items
            if self.engine.index:
                results = [
                    {"metadata": m, "index": i}
                    for i, m in enumerate(self.engine.index.metadata)
                ]
            else:
                results = []
        
        # Extract timestamps
        timed_results = []
        for r in results:
            ts = self._extract_timestamp(r.get("metadata", {}))
            if ts:
                timed_results.append((ts, r))
        
        if not timed_results:
            return []
        
        # Sort by time
        timed_results.sort(key=lambda x: x[0])
        
        # Create buckets
        buckets = self._create_buckets(timed_results, bucket_size)
        
        # Calculate trends
        buckets = self._calculate_trends(buckets)
        
        return buckets
    
    def _create_buckets(
        self,
        timed_results: List[tuple],
        bucket_size: str,
    ) -> List[TemporalBucket]:
        """Group results into time buckets."""
        if not timed_results:
            return []
        
        # Determine bucket delta
        deltas = {
            "hour": timedelta(hours=1),
            "day": timedelta(days=1),
            "week": timedelta(weeks=1),
            "month": timedelta(days=30),
            "year": timedelta(days=365),
        }
        delta = deltas.get(bucket_size, timedelta(days=1))
        
        buckets = []
        current_start = timed_results[0][0]
        current_end = current_start + delta
        current_items = []
        
        for ts, result in timed_results:
            if ts >= current_end:
                # Save current bucket
                buckets.append(TemporalBucket(
                    start=current_start,
                    end=current_end,
                    count=len(current_items),
                    items=current_items,
                ))
                # Start new bucket
                current_start = ts
                current_end = ts + delta
                current_items = []
            
            current_items.append(result)
        
        # Save last bucket
        if current_items:
            buckets.append(TemporalBucket(
                start=current_start,
                end=current_end,
                count=len(current_items),
                items=current_items,
            ))
        
        return buckets
    
    def _calculate_trends(self, buckets: List[TemporalBucket]) -> List[TemporalBucket]:
        """Calculate trend for each bucket."""
        if len(buckets) < 2:
            return buckets
        
        # Simple trend based on count change
        for i, bucket in enumerate(buckets):
            if i == 0:
                bucket.trend = "stable"
            else:
                prev_count = buckets[i-1].count
                if bucket.count > prev_count * 1.2:
                    bucket.trend = "up"
                elif bucket.count < prev_count * 0.8:
                    bucket.trend = "down"
                else:
                    bucket.trend = "stable"
        
        return buckets
    
    def find_similar_time_periods(
        self,
        reference_query: str,
        target_query: str,
        window_days: int = 7,
    ) -> List[Dict[str, Any]]:
        """Find time periods where both queries are relevant.
        
        Useful for finding correlations between concepts over time.
        
        Args:
            reference_query: First query
            target_query: Second query to correlate
            window_days: Size of time window to analyze
            
        Returns:
            List of time periods with both concepts
        """
        # Get distributions for both queries
        ref_buckets = self.temporal_distribution(reference_query, bucket_size="day")
        tgt_buckets = self.temporal_distribution(target_query, bucket_size="day")
        
        # Find overlapping high-activity periods
        overlap_periods = []
        
        for ref_bucket in ref_buckets:
            if ref_bucket.count == 0:
                continue
            
            # Look for matching target bucket
            for tgt_bucket in tgt_buckets:
                if (tgt_bucket.start <= ref_bucket.end and 
                    tgt_bucket.end >= ref_bucket.start and
                    tgt_bucket.count > 0):
                    
                    overlap_periods.append({
                        "start": max(ref_bucket.start, tgt_bucket.start),
                        "end": min(ref_bucket.end, ref_bucket.end),
                        "reference_count": ref_bucket.count,
                        "target_count": tgt_bucket.count,
                        "total": ref_bucket.count + tgt_bucket.count,
                    })
        
        # Sort by total activity
        overlap_periods.sort(key=lambda x: x["total"], reverse=True)
        
        return overlap_periods