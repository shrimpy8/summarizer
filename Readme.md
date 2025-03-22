# Document Summarizer

A Streamlit application that summarizes text, PDF, and CSV documents using LLM technology.

## Overview

This application allows users to upload documents in various formats (TXT, PDF, CSV) and generates concise summaries using language models. The summarization process consists of two steps:

1. Breaking the document into manageable chunks and summarizing each chunk
2. Creating a final comprehensive summary from all chunk summaries
3. Important to note, this is a basic app, handling a very large file is not the purpose of this app, try smaller TEXT, CSV or a PDF file. I have provided a sample text file.

## Features

- **Multiple File Format Support**: Process text, PDF, and CSV files
- **Smart Chunking**: Breaks large documents into manageable pieces
- **Two-Step Summarization**: Creates chunk summaries first, then combines them
- **Download Option**: Save the final summary as a text file

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: Document processing and LLM integration
- **LLM Options**: Supports multiple language models:
  - OpenAI (GPT-4o-mini by default)
  - Groq (Llama-3.3-70b-versatile as an alternative)

## How It Works

1. User uploads a document through the interface
2. The app processes the file based on its type
3. The document is split into smaller chunks
4. Each chunk is summarized separately
5. All chunk summaries are combined into a final comprehensive summary
6. The user can download the final summary

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via `pip`):

  ```
  streamlit
  langchain
  langchain_community
  langchain_groq
  langchain_openai
  python-dotenv
  ```

### Environment Setup

Create a `.env` file in the project directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

### Running the App

```bash
streamlit run app.py
```

## Usage

1. Launch the application
2. Upload a document (TXT, PDF, or CSV)
3. Click the "Summarize document" button
4. View the generated summary
5. Download the summary using the "Download final summary" button
