from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.embeddings import get_embeddings
from app.core.config import settings


def get_vectorstore(collection_name: str = "documents") -> Chroma:
    """
    Connects to an existing ChromaDB collection.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path
    )


def build_rag_chain(collection_name: str = "documents"):
    """
    Builds the RAG chain that connects ChromaDB with Groq.
    Retrieves relevant chunks and uses them as context for the LLM.
    """

    # Connect to vector database
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions 
    based on the provided context.
    
    If the answer is not in the context, say honestly 
    that you don't have enough information to answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """)

    # Helper to format retrieved chunks into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain using LangChain's pipe operator
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str, collection_name: str = "documents") -> dict:
    """
    Receives a question, retrieves relevant context
    and returns the LLM answer with sources.
    """

    chain, retriever = build_rag_chain(collection_name)

    # Get relevant documents for the question
    sources = retriever.invoke(question)

    # Run the chain to get the answer
    answer = chain.invoke(question)

    # Format sources to return to the user
    formatted_sources = [
        {
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "unknown")
        }
        for doc in sources
    ]

    return {
        "answer": answer,
        "sources": formatted_sources
    }