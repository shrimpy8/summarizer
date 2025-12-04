"""
Tests for document_loaders module.

Tests the DocumentProcessor class and DocxLoader functionality.
"""

import os
import sys
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_loaders import DocumentProcessor, DocxLoader, DOCX_AVAILABLE


class TestDocumentProcessor:
    """Tests for the DocumentProcessor class."""

    def test_init_default_values(self):
        """Test DocumentProcessor initializes with default chunk parameters."""
        processor = DocumentProcessor()
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 100

    def test_init_custom_values(self):
        """Test DocumentProcessor initializes with custom chunk parameters."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50

    def test_supported_types(self):
        """Test that SUPPORTED_TYPES contains expected file types."""
        processor = DocumentProcessor()
        assert "text/plain" in processor.SUPPORTED_TYPES
        assert "text/csv" in processor.SUPPORTED_TYPES
        assert "application/pdf" in processor.SUPPORTED_TYPES
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in processor.SUPPORTED_TYPES

    def test_is_supported_type_valid(self):
        """Test is_supported_type returns True for valid types."""
        processor = DocumentProcessor()
        assert processor.is_supported_type("text/plain") is True
        assert processor.is_supported_type("text/csv") is True
        assert processor.is_supported_type("application/pdf") is True

    def test_is_supported_type_invalid(self):
        """Test is_supported_type returns False for invalid types."""
        processor = DocumentProcessor()
        assert processor.is_supported_type("application/json") is False
        assert processor.is_supported_type("image/png") is False
        assert processor.is_supported_type("video/mp4") is False

    def test_get_loader_txt(self):
        """Test get_loader returns TextLoader for text/plain."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            loader = processor.get_loader(temp_path, "text/plain")
            assert loader is not None
            assert "TextLoader" in type(loader).__name__
        finally:
            os.unlink(temp_path)

    def test_get_loader_csv(self):
        """Test get_loader returns CSVLoader for text/csv."""
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1,col2\nval1,val2")
            temp_path = f.name

        try:
            loader = processor.get_loader(temp_path, "text/csv")
            assert loader is not None
            assert "CSVLoader" in type(loader).__name__
        finally:
            os.unlink(temp_path)

    def test_get_loader_pdf(self):
        """Test get_loader returns PyPDFLoader for application/pdf."""
        processor = DocumentProcessor()
        # PDF loader doesn't need actual file for instantiation
        loader = processor.get_loader("/fake/path.pdf", "application/pdf")
        assert loader is not None
        assert "PyPDFLoader" in type(loader).__name__

    def test_get_loader_unsupported(self):
        """Test get_loader raises ValueError for unsupported types."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError) as exc_info:
            processor.get_loader("/fake/path.json", "application/json")
        assert "Unsupported file type" in str(exc_info.value)

    def test_save_uploaded_file(self):
        """Test save_uploaded_file saves file to temp directory."""
        processor = DocumentProcessor()

        # Create mock uploaded file
        mock_file = Mock()
        mock_file.name = "test_document.txt"
        mock_file.getbuffer.return_value = b"test content for saving"

        temp_path, extension = processor.save_uploaded_file(mock_file)

        try:
            assert os.path.exists(temp_path)
            assert extension == ".txt"
            with open(temp_path, "rb") as f:
                content = f.read()
            assert content == b"test content for saving"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_cleanup_temp_file(self):
        """Test cleanup_temp_file removes the file."""
        processor = DocumentProcessor()

        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        assert os.path.exists(temp_path)
        processor.cleanup_temp_file(temp_path)
        assert not os.path.exists(temp_path)

    def test_cleanup_temp_file_nonexistent(self):
        """Test cleanup_temp_file handles non-existent file gracefully."""
        processor = DocumentProcessor()
        # Should not raise an error
        processor.cleanup_temp_file("/nonexistent/path/file.txt")

    def test_load_and_split_unsupported_type(self):
        """Test load_and_split raises ValueError for unsupported file types."""
        processor = DocumentProcessor()

        mock_file = Mock()
        mock_file.type = "application/json"
        mock_file.name = "test.json"

        with pytest.raises(ValueError) as exc_info:
            processor.load_and_split(mock_file)
        assert "not supported" in str(exc_info.value)


@pytest.mark.skipif(not DOCX_AVAILABLE, reason="python-docx not installed")
class TestDocxLoader:
    """Tests for the DocxLoader class."""

    def test_init(self):
        """Test DocxLoader initializes with file path."""
        loader = DocxLoader("/path/to/file.docx")
        assert loader.file_path == "/path/to/file.docx"

    def test_load_returns_documents(self):
        """Test load returns list of Document objects."""
        # Create a mock docx file
        with patch('document_loaders.DocxDocument') as mock_docx:
            mock_doc = MagicMock()
            mock_para1 = MagicMock()
            mock_para1.text = "First paragraph."
            mock_para2 = MagicMock()
            mock_para2.text = "Second paragraph."
            mock_doc.paragraphs = [mock_para1, mock_para2]
            mock_doc.tables = []
            mock_docx.return_value = mock_doc

            loader = DocxLoader("/path/to/file.docx")
            docs = loader.load()

            assert len(docs) == 1
            assert "First paragraph" in docs[0].page_content
            assert "Second paragraph" in docs[0].page_content


class TestDocxAvailability:
    """Tests for DOCX availability check."""

    def test_docx_available_flag(self):
        """Test DOCX_AVAILABLE flag is properly set."""
        # This should be True if python-docx is installed
        assert isinstance(DOCX_AVAILABLE, bool)
