# Install: pip install streamlit sentence-transformers PyMuPDF faiss-cpu requests groq

import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import requests

import numpy as np
import faiss

from groq import Groq

st.title("📚 RAG with TinyLlama/Groq ")
st.write("Upload PDF/Text → Ask Questions → Get Smart Answers!")

# Model Configuration
st.sidebar.header("⚙️ Model Settings")
model_provider = st.sidebar.selectbox(
    "Choose Model Provider:",
    ["Ollama (Local)", "Groq (Cloud)"]
)

if model_provider == "Ollama (Local)":
    st.sidebar.subheader("🦙 Ollama Settings")
    ollama_host = st.sidebar.text_input("Ollama Host:", value="http://localhost:11434")
    selected_model = "tinyllama"
    st.sidebar.write(f"Using model: **{selected_model}**")

else:  # Groq
    st.sidebar.subheader("⚡ Groq Settings")
    groq_api_key = st.sidebar.text_input("Groq API Key:", type="password")
    selected_model = st.sidebar.selectbox(
        "Choose Groq Model:",
        ["llama3-8b-8192", "llama3-70b-8192"]
    )

    if not groq_api_key:
        st.sidebar.warning("⚠️ Please enter your Groq API key")

# Test connections
def test_ollama_connection():
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return True, model_names
        return False, []
    except Exception as e:
        return False, str(e)

def test_groq_connection(api_key):
    try:
        client = Groq(api_key=api_key)
        # Test with a simple completion
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama3-8b-8192",
            max_tokens=5
        )
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

# Check connection status
with st.sidebar:
    if st.button("🔍 Test Connection"):
        if model_provider == "Ollama (Local)":
            is_connected, result = test_ollama_connection()
            if is_connected:
                st.success("✅ Ollama connected!")
                st.write("Available models:", result)
            else:
                st.error(f"❌ Ollama not accessible: {result}")
        else:  # Groq
            if groq_api_key:
                is_connected, result = test_groq_connection(groq_api_key)
                if is_connected:
                    st.success("✅ Groq connected!")
                else:
                    st.error(f"❌ Groq error: {result}")
            else:
                st.error("❌ Please enter Groq API key first")

# File Upload
uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=['pdf', 'txt']
)

def generate_ollama_response(prompt, model_name, host):
    """Generate response using Ollama API"""
    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40
            }
        }

        response = requests.post(
            f"{host}/api/generate",
            json=payload,
            timeout=120  # Longer timeout for local models
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def generate_groq_response(prompt, model_name, api_key):
    """Generate response using Groq API"""
    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.1,
            max_tokens=1000,
            top_p=0.9
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error connecting to Groq: {str(e)}"

if uploaded_file:
    # Get file info for cache key
    file_name = uploaded_file.name
    file_size = uploaded_file.size

    # Create a unique key for caching based on filename and size
    cache_key = f"{file_name}_{file_size}"

    # Store current file info in session state to detect changes
    if 'current_file_key' not in st.session_state:
        st.session_state.current_file_key = None

    # Clear cache if a different file is uploaded
    if st.session_state.current_file_key != cache_key:
        st.cache_data.clear()
        st.session_state.current_file_key = cache_key

    # STEP 1: Extract text
    @st.cache_data
    def extract_text_from_file(file_name, file_bytes):
        """Extract text from uploaded file"""
        if file_name.endswith('.pdf'):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            clean_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                clean_text += page.get_text() + "\n\n"
            doc.close()
        else:
            clean_text = str(file_bytes, "utf-8")
        return clean_text

    with st.spinner("🔍 Reading document..."):
        file_bytes = uploaded_file.read()
        clean_text = extract_text_from_file(file_name, file_bytes)

    st.success(f"✅ Extracted {len(clean_text):,} characters from **{file_name}**")

    # Show first bit of text
    with st.expander("📋 Preview of extracted text:"):
        st.text_area("First 500 characters:", clean_text[:500], height=150)

    # STEP 2: Break into chunks
    chunk_size = st.sidebar.slider("Chunk Size:", 200, 800, 400)

    @st.cache_data
    def create_chunks(text, chunk_size, file_key):
        """Create chunks from text"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if chunk.strip():  # Skip empty chunks
                chunks.append(chunk)
        return chunks

    chunks = create_chunks(clean_text, chunk_size, cache_key)
    st.write(f"📝 Created {len(chunks)} chunks from **{file_name}**")

    # STEP 3: Create embeddings
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer('BAAI/bge-large-en-v1.5')

    model = load_embedding_model()

    @st.cache_data
    def create_embeddings(_chunks, file_key):
        """Create embeddings for chunks"""
        embeddings = model.encode(_chunks, show_progress_bar=True)
        return embeddings

    with st.spinner("🧠 Converting to embeddings..."):
        embeddings = create_embeddings(chunks, cache_key)

    st.write("✅ Created embeddings")

    # STEP 4: Store in FAISS
    @st.cache_data
    def setup_faiss_database(_embeddings, _chunks, file_name, file_key):
        """Setup FAISS database with proper file identification"""
        # Convert embeddings to numpy array
        embeddings_array = np.array(_embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add embeddings to index
        index.add(embeddings_array)

        # Create metadata with file info
        metadata = []
        for i, chunk in enumerate(_chunks):
            metadata.append({
                "chunk_id": i,
                "source": file_name,
                "text": chunk,
                "file_key": file_key
            })

        return index, metadata, embeddings_array

    index, metadata, embeddings_array = setup_faiss_database(embeddings, chunks, file_name, cache_key)
    st.write(f"💾 Stored in FAISS index for **{file_name}**")

    # STEP 5: Ask Questions!
    st.header("❓ Ask Your Question")
    st.write(f"Currently querying: **{file_name}**")
    question = st.text_area("What do you want to know?", height=100)

    if question:
        # Test connection before processing
        if model_provider == "Ollama (Local)":
            is_connected, _ = test_ollama_connection()
            if not is_connected:
                st.error("❌ Ollama is not running or accessible. Please start Ollama and try again.")
                st.code("ollama serve")
                st.stop()
        else:  # Groq
            if not groq_api_key:
                st.error("❌ Please enter your Groq API key in the sidebar.")
                st.stop()

        with st.spinner("🔍 Searching for relevant information..."):
            # Convert question to embedding
            question_embedding = model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Normalize for cosine similarity
            faiss.normalize_L2(question_embedding)

            # Search index
            k = st.sidebar.slider("Number of sources:", 1, 5, 3)
            scores, indices = index.search(question_embedding, k)

            # Get relevant chunks and metadata
            relevant_chunks = []
            relevant_metadata = []
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    relevant_chunks.append(metadata[idx]["text"])
                    relevant_metadata.append({
                        "chunk_id": metadata[idx]["chunk_id"],
                        "source": metadata[idx]["source"],
                        "score": scores[0][i],
                        "file_key": metadata[idx]["file_key"]
                    })

            context = "\n\n".join(relevant_chunks)

        # Generate answer
        provider_name = "Ollama" if model_provider == "Ollama (Local)" else "Groq"
        with st.spinner(f"🤖 Generating answer with {provider_name} ({selected_model})..."):
            prompt = f"""You are an expert assistant. Answer the user's question only using the context below. If the answer is not in the context, say you don't know.


Context from document ({file_name}):
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

            if model_provider == "Ollama (Local)":
                answer = generate_ollama_response(prompt, selected_model, ollama_host)
            else:  # Groq
                answer = generate_groq_response(prompt, selected_model, groq_api_key)

        # Display answer
        st.subheader("🎯 Answer:")
        st.write(answer)

        # Show sources using better display method
        with st.expander(f"📚 Sources Used from {file_name}"):
            for i, (chunk, meta) in enumerate(zip(relevant_chunks, relevant_metadata)):
                similarity = meta["score"] * 100
                st.write(f"**Source {i+1}** (Similarity: {similarity:.1f}%)")
                st.write(f"From: {meta['source']}")
                # Use st.code for better display of read-only content
                st.code(chunk, language="text")

else:
    st.info("👆 Upload a document to get started!")
    # Clear session state when no file is uploaded
    if 'current_file_key' in st.session_state:
        del st.session_state.current_file_key