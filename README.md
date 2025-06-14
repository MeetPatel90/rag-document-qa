# RAG Document Q&A App

Upload documents and ask questions using AI.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open browser:** Go to `http://localhost:8501`

## Usage

1. Choose AI provider in sidebar:
   - **Groq**: Get free API key from [console.groq.com](https://console.groq.com)
   - **Ollama**: 
     - Install from [ollama.ai](https://ollama.ai)
     - Run `ollama serve`
     - Run `ollama pull tinyllama`
     - In app sidebar, set host to `http://localhost:11434`

2. Upload a PDF or TXT file

3. Ask questions about the document
