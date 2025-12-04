"""
Document Summarizer Application

A Streamlit-based application for summarizing text, PDF, CSV, and Word documents
using Large Language Models (LLMs) with automatic retry logic, comprehensive
error handling, progress tracking, and quality metrics.

Author: Harsh
Date: 2025-12-03
Version: 2.1.0
"""

from typing import Dict, Any, List
import time
import streamlit as st

# Page config must be first Streamlit command - enables wide layout
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",  # This makes the main content area wider
    initial_sidebar_state="expanded"
)

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
from progress_tracker import StreamlitProgressTracker
from quality_metrics import QualityAnalyzer, display_quality_metrics


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

# Custom CSS for improved UI layout
st.markdown("""
<style>
    /* Increase overall font size */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* Increase paragraph and text font size */
    p, li, span, div {
        font-size: 1.1rem !important;
    }

    /* Larger headings */
    h1 {
        font-size: 2.2rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }

    /* Increase sidebar width by 5% */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
    }

    [data-testid="stSidebarContent"] {
        width: 100% !important;
    }

    /* Reduce main content area by 5% */
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"],
    .block-container,
    [class*="block-container"] {
        max-width: 95% !important;
        width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Target the main section */
    .main,
    section.main,
    [data-testid="stMain"] {
        width: 100% !important;
    }

    /* Reduce the AppView padding/margin */
    .stApp > header + div,
    [data-testid="stAppViewContainer"] {
        padding-left: 0 !important;
    }

    /* Target inner content wrapper */
    .stApp [data-testid="stAppViewContainer"] > section {
        padding-left: 0.5rem !important;
    }

    /* Larger button text */
    .stButton > button {
        font-size: 1.1rem !important;
    }

    /* Larger selectbox text */
    .stSelectbox label, .stSelectbox div {
        font-size: 1.05rem !important;
    }

    /* Larger metrics text */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title(ui_config.get('title', 'Summarizer App'))
st.divider()

st.markdown(ui_config.get('description', '## Start summarizing your documents.'))

# ================================================================================
# SIDEBAR CONFIGURATION SELECTOR
# ================================================================================
# Allows users to switch LLM provider and model without editing config files

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")

    # LLM Provider selection
    st.subheader("LLM Provider")

    # Available providers and their models
    PROVIDER_MODELS = {
        "OpenAI": {
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "default": config.openai_model
        },
        "Groq": {
            "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
            "default": config.groq_model
        }
    }

    # Initialize session state for provider and model if not exists
    if 'selected_provider' not in st.session_state:
        # Use config default
        st.session_state.selected_provider = "OpenAI" if config.llm_provider == "openai" else "Groq"

    if 'selected_model' not in st.session_state:
        default_provider = st.session_state.selected_provider
        st.session_state.selected_model = PROVIDER_MODELS[default_provider]["default"]

    # Provider selectbox
    selected_provider = st.selectbox(
        "Select Provider",
        options=list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(st.session_state.selected_provider),
        help="Choose between OpenAI (paid) or Groq (free tier available)",
        key="provider_select"
    )

    # Update session state if provider changed
    if selected_provider != st.session_state.selected_provider:
        st.session_state.selected_provider = selected_provider
        # Reset model to provider default when switching providers
        st.session_state.selected_model = PROVIDER_MODELS[selected_provider]["default"]
        st.rerun()

    # Model selectbox (based on selected provider)
    available_models = PROVIDER_MODELS[selected_provider]["models"]
    default_model_index = (
        available_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models
        else 0
    )

    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        index=default_model_index,
        help="Select the specific model to use for summarization",
        key="model_select"
    )

    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

    # Display current selection
    st.markdown("---")
    st.caption(f"**Active:** {selected_provider} / {selected_model}")

    # Provider info
    st.markdown("---")
    st.subheader("ℹ️ Provider Info")
    if selected_provider == "OpenAI":
        st.caption(
            "OpenAI provides high-quality models. "
            "Requires OPENAI_API_KEY in .env file."
        )
    else:
        st.caption(
            "Groq offers fast inference with free tier. "
            "Requires GROQ_API_KEY in .env file."
        )

    # Advanced settings expander
    with st.expander("📐 Advanced Settings"):
        st.caption(f"Chunk Size: {config.chunk_size} chars")
        st.caption(f"Chunk Overlap: {config.chunk_overlap} chars")
        st.caption(f"Max Retries: {config.max_retry_attempts}")

# ================================================================================
# FILE UPLOAD WIDGET
# ================================================================================

file_uploader_config = ui_config.get('file_uploader', {})
uploaded_file = st.file_uploader(
    file_uploader_config.get('label', 'Upload a Text, PDF, CSV, or Word document.'),
    type=["txt", "pdf", "csv", "docx"],
    help=file_uploader_config.get('help_text', 'Select a file to process. Supported formats: TXT, PDF, CSV, DOCX')
)

# ================================================================================
# INITIALIZE COMPONENTS
# ================================================================================

# Document processor: Handles loading, chunking, and cleanup (configured from config.yaml)
doc_processor = DocumentProcessor(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap
)

# LLM Model Configuration (dynamic based on UI selection)
# Provider and model are selected via sidebar UI
selected_provider = st.session_state.get('selected_provider', 'OpenAI')
selected_model = st.session_state.get('selected_model', config.openai_model)

if selected_provider == 'Groq':
    llm = ChatGroq(model=selected_model)
    logger.info(f"Using Groq LLM: {selected_model}")
else:
    llm = ChatOpenAI(model=selected_model)
    logger.info(f"Using OpenAI LLM: {selected_model}")

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

    # Initialize progress tracker for chunk processing
    progress_tracker = StreamlitProgressTracker(
        total_items=len(chunks),
        description="Summarizing chunks"
    )

    try:
        logger.info(f"Starting to summarize {len(chunks)} chunks")

        # Start progress tracking
        progress_tracker.start()

        # Process each chunk individually
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            # Track start time for this chunk
            chunk_start_time = time.time()

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

            # Update progress tracker
            chunk_time = time.time() - chunk_start_time
            progress_tracker.update(chunk_time)

        # Mark chunk processing as complete
        progress_tracker.complete()

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

            # ====================================================================
            # QUALITY METRICS
            # ====================================================================
            # Analyze and display summary quality metrics

            # Get original document text from chunks
            original_text = "\n".join([
                chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                for chunk in chunks
            ])

            # Analyze summary quality
            quality_analyzer = QualityAnalyzer()
            quality_metrics = quality_analyzer.analyze(original_text, final_summary)

            # Display quality metrics to user
            display_quality_metrics(quality_metrics)

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



