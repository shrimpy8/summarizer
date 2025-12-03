"""
Document Summarizer Application

A Streamlit-based application for summarizing text, PDF, and CSV documents
using Large Language Models (LLMs) with automatic retry logic and comprehensive
error handling.

Author: Harsh
Date: 2025-11-30
Version: 2.0.0
"""

from typing import Dict, Any, List
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIConnectionError, APIError
from document_loaders import DocumentProcessor
from config_loader import get_config


# ================================================================================
# CONFIGURATION AND INITIALIZATION
# ================================================================================

# Load environment variables (API keys)
load_dotenv()

# Load application configuration
config = get_config()

# Configure logging from config file
log_config = config.get_logging_config()
handlers = []
if log_config.get('file_enabled', True):
    handlers.append(logging.FileHandler(log_config.get('filename', 'summarizer.log')))
if log_config.get('console_enabled', True):
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, log_config.get('level', 'INFO')),
    format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Retry decorator for LLM API calls with exponential backoff (configured from config.yaml)
@retry(
    stop=stop_after_attempt(config.max_retry_attempts),
    wait=wait_exponential(
        multiplier=config.retry_backoff_multiplier,
        min=config.retry_backoff_min,
        max=config.retry_backoff_max
    ),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    reraise=True
)
def invoke_llm_with_retry(
    chain: Any,
    input_data: Dict[str, Any],
    operation_name: str = "LLM"
) -> str:
    """
    Invoke LLM chain with automatic retry logic for transient errors.

    This function wraps LangChain chain invocations with exponential backoff retry logic
    to handle transient API errors. It will retry up to 3 times with increasing wait
    times (2, 4, 8 seconds) before giving up.

    Args:
        chain: LangChain chain to invoke (typically prompt | llm | parser)
        input_data: Input dictionary for the chain (e.g., {"document": doc_content})
        operation_name: Human-readable name for logging purposes (default: "LLM")

    Returns:
        str: The LLM's response as a string

    Raises:
        RateLimitError: If API rate limit is exceeded after all retry attempts
        APIConnectionError: If connection fails after all retry attempts
        APIError: If API error occurs after all retry attempts

    Example:
        >>> chain = prompt_template | llm | parser
        >>> result = invoke_llm_with_retry(chain, {"document": "text"}, "Summary")
        >>> print(result)
    """
    try:
        logger.info(f"Invoking {operation_name}...")
        response = chain.invoke(input_data)
        logger.info(f"{operation_name} completed successfully")
        return response
    except RateLimitError as e:
        logger.warning(f"Rate limit hit for {operation_name}, retrying...")
        raise
    except APIConnectionError as e:
        logger.warning(f"Connection error for {operation_name}, retrying...")
        raise
    except APIError as e:
        logger.warning(f"API error for {operation_name}, retrying...")
        raise

# ================================================================================
# STREAMLIT UI CONFIGURATION
# ================================================================================

ui_config = config.get_ui_config()

st.title(ui_config.get('title', 'Summarizer App'))
st.divider()

st.markdown(ui_config.get('description', '## Start summarizing your documents.'))

# ================================================================================
# FILE UPLOAD WIDGET
# ================================================================================

file_uploader_config = ui_config.get('file_uploader', {})
uploaded_file = st.file_uploader(
    file_uploader_config.get('label', 'Upload a Text, PDF, or CSV file.'),
    type=["txt", "pdf", "csv"],
    help=file_uploader_config.get('help_text', 'Select a file to process. Supported formats: TXT, PDF, CSV')
)

# ================================================================================
# INITIALIZE COMPONENTS
# ================================================================================

# Document processor: Handles loading, chunking, and cleanup (configured from config.yaml)
doc_processor = DocumentProcessor(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap
)

# LLM Model Configuration (configured from config.yaml)
# Provider can be changed in config.yaml: 'openai' or 'groq'
if config.llm_provider == 'groq':
    llm = ChatGroq(model=config.groq_model)
    logger.info(f"Using Groq LLM: {config.groq_model}")
else:
    llm = ChatOpenAI(model=config.openai_model)
    logger.info(f"Using OpenAI LLM: {config.openai_model}")

# Output parser: Converts LLM response to string
parser = StrOutputParser()

# Basic prompt template (not used in main flow, kept for reference)
prompt_template = ChatPromptTemplate.from_template("Summarize the following document: {document}")

# Create a basic chain (not used in main flow)
chain = prompt_template | llm | parser

# ================================================================================
# FILE UPLOAD AND PROCESSING
# ================================================================================

if uploaded_file is not None:
    messages = ui_config.get('messages', {})
    with st.spinner(messages.get('processing_file', 'Processing the uploaded file...')):
        try:
            # Display processing information to user
            st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.type})")
            logger.info(f"Processing file: {uploaded_file.name}, type: {uploaded_file.type}")

            # Load document and split into chunks using DocumentProcessor
            # This handles: file validation, temp file management, loading, chunking, cleanup
            chunks = doc_processor.load_and_split(uploaded_file)

            logger.info(f"Successfully processed {uploaded_file.name} into {len(chunks)} chunks")

        except ValueError as e:
            # Handle unsupported file type errors
            error_messages = ui_config.get('errors', {})
            logger.error(f"Unsupported file type: {str(e)}", exc_info=True)
            st.error(error_messages.get('unsupported_file', '❌ {error}').format(error=str(e)))
            st.stop()
        except Exception as e:
            # Handle all other file processing errors
            error_messages = ui_config.get('errors', {})
            logger.error(f"File processing error: {str(e)}", exc_info=True)
            st.error(error_messages.get('processing_error', '❌ An error occurred while processing the file: {error}').format(error=str(e)))
            st.stop()

    success_messages = ui_config.get('success', {})
    st.success(success_messages.get('file_uploaded', '✅ File uploaded and processed into {count} chunks').format(count=len(chunks)))

# ================================================================================
# SUMMARIZATION WORKFLOW
# ================================================================================

if st.button("Summarize document"):

    # Container for final summary output
    container = st.empty()

    # ============================================================================
    # STEP 1: CHUNK-LEVEL SUMMARIZATION
    # ============================================================================
    # Summarize each document chunk individually to handle large documents
    # that exceed token limits

    chunk_summaries: List[str] = []

    prompts = config.get_prompts()
    with st.spinner(messages.get('summarizing_chunks', 'Summarizing chunks...')):
        try:
            logger.info(f"Starting to summarize {len(chunks)} chunks")

            # Process each chunk individually
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                # Create prompt for chunk summarization from config
                # Template loaded from config.yaml for easy customization
                chunk_prompt = ChatPromptTemplate.from_template(
                    prompts.get('chunk_summary', config.chunk_prompt_template)
                )

                # Build LangChain: prompt → LLM → string parser
                chunk_chain = chunk_prompt | llm | parser

                # Invoke with automatic retry logic (3 attempts, exponential backoff)
                chunk_summary = invoke_llm_with_retry(
                    chunk_chain,
                    {"document": chunk},
                    f"Chunk {i+1}/{len(chunks)}"
                )
                chunk_summaries.append(chunk_summary)

        except RateLimitError as e:
            # API rate limit hit even after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"Rate limit exceeded after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('rate_limit', '⚠️ API rate limit exceeded. Please wait a moment and try again.'))
            st.stop()
        except APIConnectionError as e:
            # Network/connection issues after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"Connection error after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('connection_error', '⚠️ Unable to connect to the AI service. Please check your internet connection.'))
            st.stop()
        except APIError as e:
            # General API errors after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"API error after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('api_error', '⚠️ AI service error: {error}').format(error=str(e)))
            st.stop()
        except Exception as e:
            # Catch-all for unexpected errors
            error_messages = ui_config.get('errors', {})
            logger.error(f"Unexpected error summarizing chunks: {str(e)}", exc_info=True)
            st.error(error_messages.get('unexpected_error', '❌ An unexpected error occurred: {error}').format(error=str(e)))
            st.stop()

    # ============================================================================
    # STEP 2: FINAL SUMMARY GENERATION
    # ============================================================================
    # Combine all chunk summaries into a single, cohesive final summary

    with st.spinner(messages.get('creating_final_summary', 'Creating final summary from document...')):
        try:
            # Combine all chunk summaries with newlines
            combined_summaries = "\n".join(chunk_summaries)

            # Create prompt for final summarization from config
            # Template loaded from config.yaml for easy customization
            final_prompt = ChatPromptTemplate.from_template(
                prompts.get('final_summary', config.final_prompt_template)
            )

            # Build final LangChain: prompt → LLM → string parser
            final_chain = final_prompt | llm | parser

            # Invoke with automatic retry logic (3 attempts, exponential backoff)
            final_summary = invoke_llm_with_retry(
                final_chain,
                {"document": combined_summaries},
                "Final Summary"
            )

            # Log success and display summary to user
            logger.info(f"Final summary generated successfully ({len(final_summary)} characters)")
            container.write(final_summary)

            # ====================================================================
            # DOWNLOAD BUTTON
            # ====================================================================
            # Allow user to export the summary as a text file

            download_config = ui_config.get('download_button', {})
            st.download_button(
                label=download_config.get('label', 'Download final summary'),
                data=final_summary,
                file_name=download_config.get('filename', 'final_summary.txt'),
                mime="text/plain"
            )

        except RateLimitError as e:
            # API rate limit hit even after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"Rate limit exceeded after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('rate_limit', '⚠️ API rate limit exceeded. Please wait a moment and try again.'))
        except APIConnectionError as e:
            # Network/connection issues after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"Connection error after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('connection_error', '⚠️ Unable to connect to the AI service. Please check your internet connection.'))
        except APIError as e:
            # General API errors after retries
            error_messages = ui_config.get('errors', {})
            logger.error(f"API error after retries: {str(e)}", exc_info=True)
            st.error(error_messages.get('api_error', '⚠️ AI service error: {error}').format(error=str(e)))
        except Exception as e:
            # Catch-all for unexpected errors
            error_messages = ui_config.get('errors', {})
            logger.error(f"Unexpected error creating final summary: {str(e)}", exc_info=True)
            st.error(error_messages.get('unexpected_error', '❌ An unexpected error occurred: {error}').format(error=str(e)))



