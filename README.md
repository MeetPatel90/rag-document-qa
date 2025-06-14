# RAG Document Q&A App

Upload documents and ask questions using AI.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. Install OLlama and get Groq API Key:
   - **Groq**: Get free API key from [console.groq.com](https://console.groq.com)
   - **Ollama**: 
     - Install from [ollama.ai](https://ollama.ai)(change DNS if installing via Homebrew and get error)
     - Run `ollama serve`(Keep this terminal open)
     - Run `ollama pull tinyllama`(Download tinyllama in another terminal)
     - In app sidebar, set host to `http://localhost:11434`(by default, change if not)

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open browser:** Go to `http://localhost:8501`

## Usage

1. Choose AI provider in sidebar:
   - **Groq**: Add API key and choose Model, then use Test.  
   - **Ollama**: 
     - In app sidebar, set host to `http://localhost:11434`(by default, change if not), then use Test

2. Upload a PDF or TXT file

3. Ask questions about the document
