from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns the embedding model.
    Uses HuggingFace because it runs locally for free.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )