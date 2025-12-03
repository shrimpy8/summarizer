# Document Summarizer

A Streamlit application that intelligently summarizes text, PDF, and CSV documents using advanced LLM technology with built-in retry logic, structured logging, and comprehensive error handling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)](https://langchain.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Reliability & Error Handling](#reliability--error-handling)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Technologies](#technologies)
- [Limitations](#limitations)

---

## 🎯 Overview

This application provides intelligent document summarization through a user-friendly web interface. It processes documents by breaking them into manageable chunks, summarizing each chunk individually, and then combining these summaries into a cohesive final summary.

**Key Highlights**:
- 🚀 **Production-Ready**: Built with retry logic, comprehensive error handling, and structured logging
- 📁 **Multi-Format Support**: Handles TXT, PDF, and CSV files seamlessly
- 🔄 **Two-Stage Summarization**: Chunk-level then document-level for better quality
- 💾 **Export Capability**: Download summaries as text files
- ⚡ **Flexible LLM Backend**: Supports OpenAI GPT and Groq Llama models

---

## ✨ Features

### Core Functionality
- **Multiple File Format Support**: Process text (.txt), PDF (.pdf), and CSV (.csv) files
- **Smart Document Chunking**: Automatically splits large documents into optimal-sized chunks (1000 chars with 100 char overlap)
- **Two-Step Summarization**:
  1. Generate summaries for each document chunk
  2. Combine chunk summaries into a comprehensive final summary
- **Download Capability**: Export final summaries as `.txt` files

### Reliability & Production Features
- **Automatic Retry Logic**: Exponential backoff for API failures (3 attempts, 2-10 second wait)
- **Comprehensive Error Handling**: Graceful handling of rate limits, connection errors, and API failures
- **Structured Logging**: Detailed logs to both file (`summarizer.log`) and console
- **User-Friendly Error Messages**: Clear, actionable feedback for all error conditions
- **Modular Architecture**: Separated document processing logic for maintainability

---

## 🏗️ Architecture

### Summarization Pipeline

```
User Upload → Document Processing → Chunking → Chunk Summarization → Final Summary → Download
```

**Detailed Flow**:
1. **Upload**: User selects a document (TXT/PDF/CSV)
2. **Validation**: File type verification and MIME type checking
3. **Processing**: Document loaded using appropriate loader (TextLoader/PyPDFLoader/CSVLoader)
4. **Chunking**: Document split into 1000-character chunks with 100-character overlap
5. **Chunk Summarization**: Each chunk summarized individually with retry logic
6. **Final Summarization**: All chunk summaries combined into cohesive summary
7. **Export**: User can download the final summary

### Retry Mechanism

```python
# Automatic retry with exponential backoff
- Attempt 1: Immediate
- Attempt 2: Wait 2 seconds
- Attempt 3: Wait 4 seconds
- Failure: Clear error message to user
```

---

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **API Keys**: At least one of the following:
  - OpenAI API key (recommended)
  - Groq API key (free alternative)
- **Internet Connection**: Required for LLM API calls

---

## 🚀 Installation

### Step 1: Clone or Download

```bash
cd /path/to/your/projects
# If cloning from a repository:
git clone <repository-url>
cd summarizer
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n summarizer python=3.8
conda activate summarizer
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages**:
- `streamlit` - Web interface framework
- `langchain` - LLM orchestration framework
- `langchain-core` - Core LangChain components
- `langchain-community` - Community document loaders
- `langchain-groq` - Groq LLM integration
- `langchain-openai` - OpenAI LLM integration
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management
- `tenacity` - Retry logic with exponential backoff

---

## ⚙️ Configuration

### Environment Variables Setup

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys**:
   ```bash
   # OpenAI Configuration (Primary)
   # Get your API key from: https://platform.openai.com/api-keys
   OPENAI_API_KEY=sk-proj-...your-actual-key-here...

   # Groq Configuration (Optional - Free Alternative)
   # Get your API key from: https://console.groq.com/keys
   GROQ_API_KEY=gsk_...your-actual-key-here...
   ```

### Getting API Keys

**OpenAI API Key**:
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy and paste into `.env` file

**Groq API Key** (Free Alternative):
1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Create a new API key
4. Copy and paste into `.env` file

### YAML Configuration

All application settings are centralized in `config.yaml`:

```yaml
# Model selection (openai or groq)
llm:
  provider: "openai"
  openai_model: "gpt-4o-mini"
  groq_model: "llama-3.3-70b-versatile"

# Retry configuration
retry:
  max_attempts: 3
  min_wait_seconds: 2
  max_wait_seconds: 10

# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "summarizer.log"
```

### Switching LLM Models

**Option 1**: Edit `config.yaml` (recommended):
```yaml
llm:
  provider: "groq"  # Change from "openai" to "groq"
```

**Option 2**: Edit `app.py` (lines 82-83):
```python
# OpenAI (default)
llm = ChatOpenAI(model="gpt-4o-mini")

# Groq (free alternative - uncomment to use)
# llm = ChatGroq(model="llama-3.3-70b-versatile")
```

---

## 💻 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Summarizer

1. **Upload Document**:
   - Click "Browse files" or drag and drop
   - Select a `.txt`, `.pdf`, or `.csv` file
   - Wait for "File uploaded and processed" message

2. **Generate Summary**:
   - Click "Summarize document" button
   - Watch the progress spinner as chunks are processed
   - View the generated summary in the interface

3. **Download Summary**:
   - Click "Download final summary" button
   - Save as `final_summary.txt` or rename as desired

### Example Workflow

```bash
# 1. Start the app
streamlit run app.py

# 2. Upload your document (e.g., research_paper.pdf)
# 3. Click "Summarize document"
# 4. Wait for processing (time depends on document size)
# 5. Review the summary
# 6. Click "Download final summary" to save
```

---

## 📁 Project Structure

```
summarizer/
├── app.py                    # Main Streamlit application
├── document_loaders.py       # Document processing module
├── config_loader.py          # Configuration management (singleton pattern)
├── config.yaml              # YAML configuration for all settings
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variable template
├── .env                     # Your API keys (DO NOT COMMIT)
├── .gitignore              # Git ignore rules
├── summarizer.log          # Application logs (auto-generated)
└── README.md               # This file
```

### Key Files

**`app.py`** (327 lines):
- Main application logic
- Streamlit UI components
- LLM chain configuration
- Retry logic and error handling

**`document_loaders.py`** (158 lines):
- `DocumentProcessor` class
- File type validation
- Document loading and chunking
- Temporary file management

**`config_loader.py`** (187 lines):
- `ConfigLoader` singleton class
- YAML configuration management
- Type-safe configuration access
- Validation and error handling

**`config.yaml`** (135 lines):
- Centralized configuration for all settings
- LLM model parameters
- Retry logic configuration
- Logging configuration
- Error message templates

**`requirements.txt`**:
- All Python package dependencies
- PyYAML for configuration management
- Pin versions for reproducibility

**`.env.example`**:
- Template for environment variables
- Safe to commit to version control

---

## 🛡️ Reliability & Error Handling

### Retry Logic

The application uses **automatic retry with exponential backoff** for all LLM API calls:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError))
)
```

**Retries on**:
- `RateLimitError`: API rate limits exceeded
- `APIConnectionError`: Network connectivity issues
- `APIError`: General API errors

**Does NOT retry**:
- Invalid API keys (permanent error)
- Malformed requests (code bug)
- Unsupported file types (user error)

### Error Messages

| Error Type | User Message | Suggested Action |
|------------|--------------|------------------|
| Rate Limit | ⚠️ API rate limit exceeded | Wait a moment and try again |
| Connection Error | ⚠️ Unable to connect to AI service | Check internet connection |
| API Error | ⚠️ AI service error | Check API key, try again later |
| Unsupported File | ❌ File type not supported | Use TXT, PDF, or CSV file |
| Processing Error | ❌ Error processing file | Check file integrity, try different file |

---

## 📊 Logging

### Log File Location

All logs are written to: **`summarizer.log`** (in project directory)

### Log Format

```
2025-11-30 14:23:45 - __main__ - INFO - Processing file: document.pdf, type: application/pdf
2025-11-30 14:23:46 - document_loaders - INFO - Using PyPDFLoader for PDF file
2025-11-30 14:23:48 - document_loaders - INFO - Created 15 chunks from document
2025-11-30 14:23:50 - __main__ - INFO - Invoking Chunk 1/15...
2025-11-30 14:23:52 - __main__ - INFO - Chunk 1/15 completed successfully
```

### Log Levels

- **INFO**: Normal operations (file uploads, processing steps, successful completions)
- **WARNING**: Retries, non-critical issues (rate limit warnings before retry)
- **ERROR**: Failures after all retries, exceptions with full stack traces

### Viewing Logs

```bash
# View entire log
cat summarizer.log

# View last 50 lines
tail -n 50 summarizer.log

# Follow logs in real-time
tail -f summarizer.log
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Module not found" errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify virtual environment is activated
which python  # Should show path to venv
```

#### 2. "Invalid API key" error

```bash
# Check .env file exists
ls -la .env

# Verify API key format
cat .env  # Should see OPENAI_API_KEY=sk-...

# Ensure no extra spaces or quotes
OPENAI_API_KEY=sk-xxxxx  # Correct
OPENAI_API_KEY = "sk-xxxxx"  # Incorrect
```

#### 3. "Rate limit exceeded" even after retries

**Solution**: You've hit your API quota
- Wait 60 seconds and try again
- Check your OpenAI/Groq dashboard for usage limits
- Consider upgrading your API plan
- Try switching to Groq (free tier has generous limits)

#### 4. PDF processing fails

**Possible Causes**:
- Scanned PDFs (images, not text)
- Encrypted/password-protected PDFs
- Corrupted file

**Solution**:
- Use OCR to extract text first
- Remove password protection
- Try a different PDF

#### 5. Streamlit won't start

```bash
# Check if port 8501 is already in use
lsof -i :8501

# Kill existing process if needed
kill -9 <PID>

# Or use a different port
streamlit run app.py --server.port 8502
```

---

## 🛠️ Technologies

### Core Framework
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[LangChain](https://python.langchain.com/)** - LLM orchestration and document processing

### LLM Providers
- **[OpenAI](https://platform.openai.com/)** - GPT-4o-mini (default)
- **[Groq](https://groq.com/)** - Llama-3.3-70b-versatile (alternative)

### Document Processing
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF parsing and text extraction
- **LangChain Community Loaders** - Text and CSV loading

### Utilities
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** - Environment variable management
- **[tenacity](https://github.com/jd/tenacity)** - Retry logic with exponential backoff

---

## ⚠️ Limitations

1. **File Size**: Optimized for documents up to ~50 pages or 100KB text
   - Larger documents may exceed API token limits
   - Processing time increases linearly with document size

2. **File Types**:
   - **PDFs**: Text-based only (not scanned images)
   - **CSVs**: Treated as plain text, not structured data analysis
   - **Images**: Not supported (no OCR)

3. **API Costs**:
   - OpenAI charges per token (input + output)
   - Groq is free but has rate limits
   - Large documents can incur significant costs

4. **Internet Required**:
   - All LLM processing happens via API calls
   - No offline mode available

5. **Summary Quality**:
   - Dependent on LLM model capabilities
   - Very technical or domain-specific content may need specialized models

---

## 📝 Sample Test File

A sample text file is included in the repository for testing. You can also use your own documents.

**Recommended test files**:
- Short articles (1-5 pages)
- Research paper abstracts
- Meeting notes
- Product documentation

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for DOCX files
- [ ] Implement custom chunk size configuration via UI
- [ ] Add support for multiple LLM model selection via dropdown
- [ ] Implement streaming responses for real-time feedback
- [ ] Add progress bar with estimated time remaining
- [ ] Support for multiple file uploads in batch

---

## 📄 License

This project is provided as-is for educational and personal use.

---

## 🆘 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `summarizer.log`
3. Verify API keys and internet connection
4. Open an issue in the repository (if applicable)

---

**Built with ❤️ using Streamlit and LangChain**
