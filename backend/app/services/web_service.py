import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.core.embeddings import get_embeddings
from app.core.config import settings


def process_url(url: str, collection_name: str = "documents") -> dict:
    """
    Receives a URL, extracts its text content
    and stores the embeddings in ChromaDB.
    """

    # Fetch the page content
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Extract clean text using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles and other non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Get clean text
    text = soup.get_text(separator="\n", strip=True)

    # Wrap text in a Document object with metadata
    doc = Document(
        page_content=text,
        metadata={"source": url, "type": "web"}
    )

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents([doc])

    # Store chunks as vectors in ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path
    )
    vectorstore.add_documents(chunks)

    return {
        "url": url,
        "chunks": len(chunks),
        "collection": collection_name
    }