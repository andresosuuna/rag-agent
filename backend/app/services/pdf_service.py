from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.core.embeddings import get_embeddings
from app.core.config import settings

def process_pdf(file_path: str, collection_name: str = "documents") -> dict:
    """
    Loads a PDF, splits it into chunks
    and stores the embeddings in ChromaDB.
    """

    # Load PDF pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Split text into chunks with overlap to avoid losing context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)

    # Store chunks as vectors in ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path
    )
    vectorstore.add_documents(chunks)

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "collection": collection_name
    }