"""
Tests for config_loader module.

Tests the Config class and get_config singleton function.
"""

import os
import sys
import tempfile
import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import Config, get_config


class TestConfig:
    """Tests for the Config class."""

    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file for testing."""
        config_data = {
            'document_processing': {
                'chunk_size': 2000,
                'chunk_overlap': 200
            },
            'llm': {
                'provider': 'openai',
                'openai': {
                    'model': 'gpt-4o-mini'
                },
                'groq': {
                    'model': 'llama-3.3-70b-versatile'
                }
            },
            'retry': {
                'max_attempts': 5,
                'backoff': {
                    'multiplier': 2,
                    'min_seconds': 1,
                    'max_seconds': 30
                }
            },
            'logging': {
                'level': 'DEBUG',
                'filename': 'test.log'
            },
            'prompts': {
                'chunk_summary': 'Summarize: {document}',
                'final_summary': 'Final: {document}'
            },
            'ui': {
                'title': 'Test App'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_init_loads_config(self, sample_config_file):
        """Test Config constructor loads YAML file."""
        config = Config(sample_config_file)
        assert config._config is not None
        assert isinstance(config._config, dict)

    def test_init_file_not_found(self):
        """Test Config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_get_simple_key(self, sample_config_file):
        """Test get method with simple key path."""
        config = Config(sample_config_file)
        provider = config.get('llm.provider')
        assert provider == 'openai'

    def test_get_nested_key(self, sample_config_file):
        """Test get method with nested key path."""
        config = Config(sample_config_file)
        model = config.get('llm.openai.model')
        assert model == 'gpt-4o-mini'

    def test_get_with_default(self, sample_config_file):
        """Test get method returns default for missing key."""
        config = Config(sample_config_file)
        value = config.get('nonexistent.key', 'default_value')
        assert value == 'default_value'

    def test_get_document_processing(self, sample_config_file):
        """Test get_document_processing returns correct section."""
        config = Config(sample_config_file)
        doc_config = config.get_document_processing()
        assert doc_config['chunk_size'] == 2000
        assert doc_config['chunk_overlap'] == 200

    def test_get_llm_config(self, sample_config_file):
        """Test get_llm_config returns correct section."""
        config = Config(sample_config_file)
        llm_config = config.get_llm_config()
        assert llm_config['provider'] == 'openai'
        assert 'openai' in llm_config
        assert 'groq' in llm_config

    def test_get_retry_config(self, sample_config_file):
        """Test get_retry_config returns correct section."""
        config = Config(sample_config_file)
        retry_config = config.get_retry_config()
        assert retry_config['max_attempts'] == 5
        assert retry_config['backoff']['multiplier'] == 2

    def test_get_logging_config(self, sample_config_file):
        """Test get_logging_config returns correct section."""
        config = Config(sample_config_file)
        log_config = config.get_logging_config()
        assert log_config['level'] == 'DEBUG'
        assert log_config['filename'] == 'test.log'

    def test_get_prompts(self, sample_config_file):
        """Test get_prompts returns correct section."""
        config = Config(sample_config_file)
        prompts = config.get_prompts()
        assert 'chunk_summary' in prompts
        assert 'final_summary' in prompts

    def test_get_ui_config(self, sample_config_file):
        """Test get_ui_config returns correct section."""
        config = Config(sample_config_file)
        ui_config = config.get_ui_config()
        assert ui_config['title'] == 'Test App'

    def test_chunk_size_property(self, sample_config_file):
        """Test chunk_size property returns correct value."""
        config = Config(sample_config_file)
        assert config.chunk_size == 2000

    def test_chunk_overlap_property(self, sample_config_file):
        """Test chunk_overlap property returns correct value."""
        config = Config(sample_config_file)
        assert config.chunk_overlap == 200

    def test_llm_provider_property(self, sample_config_file):
        """Test llm_provider property returns correct value."""
        config = Config(sample_config_file)
        assert config.llm_provider == 'openai'

    def test_openai_model_property(self, sample_config_file):
        """Test openai_model property returns correct value."""
        config = Config(sample_config_file)
        assert config.openai_model == 'gpt-4o-mini'

    def test_groq_model_property(self, sample_config_file):
        """Test groq_model property returns correct value."""
        config = Config(sample_config_file)
        assert config.groq_model == 'llama-3.3-70b-versatile'

    def test_max_retry_attempts_property(self, sample_config_file):
        """Test max_retry_attempts property returns correct value."""
        config = Config(sample_config_file)
        assert config.max_retry_attempts == 5

    def test_retry_backoff_properties(self, sample_config_file):
        """Test retry backoff properties return correct values."""
        config = Config(sample_config_file)
        assert config.retry_backoff_multiplier == 2
        assert config.retry_backoff_min == 1
        assert config.retry_backoff_max == 30

    def test_log_level_property(self, sample_config_file):
        """Test log_level property returns correct value."""
        config = Config(sample_config_file)
        assert config.log_level == 'DEBUG'

    def test_log_filename_property(self, sample_config_file):
        """Test log_filename property returns correct value."""
        config = Config(sample_config_file)
        assert config.log_filename == 'test.log'


class TestGetConfig:
    """Tests for the get_config singleton function."""

    def test_get_config_returns_config_instance(self):
        """Test get_config returns a Config instance."""
        # Reset singleton for testing
        import config_loader
        config_loader._config_instance = None

        # Use actual config file
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config.yaml'
        )

        if os.path.exists(config_path):
            config = get_config(config_path)
            assert isinstance(config, Config)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same singleton instance."""
        # Reset singleton for testing
        import config_loader
        config_loader._config_instance = None

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config.yaml'
        )

        if os.path.exists(config_path):
            config1 = get_config(config_path)
            config2 = get_config(config_path)
            assert config1 is config2
