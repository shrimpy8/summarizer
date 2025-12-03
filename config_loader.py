"""
Configuration loader module for the Document Summarizer application.

Handles loading and validating configuration from config.yaml file.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the summarizer application."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'llm.openai.model')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> model = config.get('llm.openai.model')
            >>> chunk_size = config.get('document_processing.chunk_size', 1000)
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_document_processing(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        return self._config.get('document_processing', {})

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get('llm', {})

    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry logic configuration."""
        return self._config.get('retry', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})

    def get_prompts(self) -> Dict[str, str]:
        """Get prompt templates."""
        return self._config.get('prompts', {})

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get('ui', {})

    @property
    def chunk_size(self) -> int:
        """Get chunk size for document processing."""
        return self.get('document_processing.chunk_size', 1000)

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap for document processing."""
        return self.get('document_processing.chunk_overlap', 100)

    @property
    def llm_provider(self) -> str:
        """Get LLM provider (openai or groq)."""
        return self.get('llm.provider', 'openai')

    @property
    def openai_model(self) -> str:
        """Get OpenAI model name."""
        return self.get('llm.openai.model', 'gpt-4o-mini')

    @property
    def groq_model(self) -> str:
        """Get Groq model name."""
        return self.get('llm.groq.model', 'llama-3.3-70b-versatile')

    @property
    def max_retry_attempts(self) -> int:
        """Get maximum number of retry attempts."""
        return self.get('retry.max_attempts', 3)

    @property
    def retry_backoff_multiplier(self) -> int:
        """Get retry backoff multiplier."""
        return self.get('retry.backoff.multiplier', 1)

    @property
    def retry_backoff_min(self) -> int:
        """Get minimum retry backoff seconds."""
        return self.get('retry.backoff.min_seconds', 2)

    @property
    def retry_backoff_max(self) -> int:
        """Get maximum retry backoff seconds."""
        return self.get('retry.backoff.max_seconds', 10)

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.get('logging.level', 'INFO')

    @property
    def log_filename(self) -> str:
        """Get log filename."""
        return self.get('logging.filename', 'summarizer.log')

    @property
    def chunk_prompt_template(self) -> str:
        """Get chunk summarization prompt template."""
        return self.get('prompts.chunk_summary', '')

    @property
    def final_prompt_template(self) -> str:
        """Get final summarization prompt template."""
        return self.get('prompts.final_summary', '')


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get singleton configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance
