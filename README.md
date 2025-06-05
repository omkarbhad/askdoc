# AskDoc - Document Q&A with RAG

AskDoc is a powerful document-based question-answering application that uses Retrieval-Augmented Generation (RAG) to provide accurate answers from your documents. Built with Streamlit and powered by Ollama's language models, it allows you to upload documents and ask questions in natural language.


## Features

- 📄 Upload various document formats (PDF, DOCX, TXT, HTML, MD, CSV)
- 🔍 Web URL content extraction
- 💬 Interactive chat interface with streaming responses
- 🧠 Context-aware responses using RAG
- 🎨 Modern, responsive UI with dark/light mode support
- ⚡ Fast and efficient document processing

## Prerequisites

- Python 3.8+
- Ollama server running locally (default: http://localhost:11434)
- Required Python packages (see [Installation](#installation))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/askdoc.git
   cd askdoc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama server (if not already running):
   ```bash
   ollama serve
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run ragchat.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to:
   - Select a model
   - Upload documents
   - Enter a URL to scrape content

4. Start chatting with your documents!

## Supported Models

AskDoc works with any model supported by Ollama. Some recommended models:

- `llama3`
- `mistral`
- `gemma`
- `mixtral`

## Configuration

You can configure the following environment variables in a `.env` file:

```
OLLAMA_API=http://localhost:11434
LOG_FILE=chat_log.json
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Ollama](https://ollama.ai/) for the LLM backend
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Sentence Transformers](https://www.sbert.net/) for document embeddings