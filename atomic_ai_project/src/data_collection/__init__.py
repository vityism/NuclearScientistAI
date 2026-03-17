"""
Data collection module for fetching atomic/nuclear data from IAEA API.
"""
from .iaea_client import IAEAClient
from .data_fetcher import DataFetcher

__all__ = ["IAEAClient", "DataFetcher"]