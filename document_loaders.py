"""
Document loader module for handling various file formats.

Supports: TXT, PDF, CSV
"""

import os
import tempfile
import uuid
import logging
from typing import List, Tuple
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and processing for various file formats."""

    SUPPORTED_TYPES = {
        "text/plain": "TXT",
        "text/csv": "CSV",
        "application/pdf": "PDF"
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the DocumentProcessor.

        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def is_supported_type(self, mime_type: str) -> bool:
        """
        Check if the file type is supported.

        Args:
            mime_type: MIME type of the file

        Returns:
            True if supported, False otherwise
        """
        return mime_type in self.SUPPORTED_TYPES

    def get_loader(self, file_path: str, mime_type: str):
        """
        Get the appropriate document loader for the file type.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            Document loader instance

        Raises:
            ValueError: If file type is not supported
        """
        if mime_type == "text/plain":
            logger.info("Using TextLoader for TXT file")
            return TextLoader(file_path)
        elif mime_type == "text/csv":
            logger.info("Using CSVLoader for CSV file")
            return CSVLoader(file_path)
        elif mime_type == "application/pdf":
            logger.info("Using PyPDFLoader for PDF file")
            return PyPDFLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    def save_uploaded_file(self, uploaded_file) -> Tuple[str, str]:
        """
        Save uploaded file to a temporary location.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Tuple of (temp_file_path, file_extension)
        """
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")

        logger.info(f"Saving uploaded file: {uploaded_file.name} to {temp_file_path}")

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return temp_file_path, file_extension

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Remove temporary file.

        Args:
            file_path: Path to the temporary file
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")

    def load_and_split(self, uploaded_file) -> List[Document]:
        """
        Load and split a document into chunks.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            List of document chunks

        Raises:
            ValueError: If file type is not supported
            Exception: If file processing fails
        """
        temp_file_path = None

        try:
            # Validate file type
            if not self.is_supported_type(uploaded_file.type):
                supported = ", ".join(self.SUPPORTED_TYPES.values())
                raise ValueError(
                    f"File type '{uploaded_file.type}' is not supported. "
                    f"Supported formats: {supported}"
                )

            # Save uploaded file
            temp_file_path, _ = self.save_uploaded_file(uploaded_file)

            # Get appropriate loader
            loader = self.get_loader(temp_file_path, uploaded_file.type)

            # Load document
            logger.info(f"Loading document: {uploaded_file.name}")
            doc = loader.load()

            # Split into chunks
            logger.info(f"Splitting document into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
            chunks = self.text_splitter.split_documents(doc)
            logger.info(f"Created {len(chunks)} chunks from document")

            return chunks

        finally:
            # Always cleanup temporary file
            if temp_file_path:
                self.cleanup_temp_file(temp_file_path)
