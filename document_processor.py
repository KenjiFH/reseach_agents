import tempfile
import os
# Swapped out PyPDFLoader for the much more robust PyMuPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"

def process_document_and_create_vdb(uploaded_file):
    """
    Takes a Streamlit UploadedFile, processes it with PyMuPDF, and returns a Chroma retriever.
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        print("Loading document with PyMuPDFLoader...")
        
        # 1. The Engine Swap
        # PyMuPDF parses the document layout, preserving the order of text blocks in slides
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()

        # 2. Split the Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")

        # 3. Initialize Ollama Embeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 4. Create and Populate the Chroma Vector Store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="poc_rag_collection"
            # persist_directory="./chroma_db"  # Uncomment if you want to save to disk
        )
        
        return vector_store.as_retriever(search_kwargs={"k": 4})

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)