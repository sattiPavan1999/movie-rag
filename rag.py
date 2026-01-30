from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import dotenv

# Load environment variables before initializing OpenAI client
dotenv.load_dotenv()

from openai import OpenAI
client = OpenAI()

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "movies"


def get_related_chunks(question: str, k: int = 2):
    """
    Retrieve top-k related chunks from ChromaDB
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    results = vector_store.similarity_search(
        query=question,
        k=k
    )

    return results


def create_prompt(retrieved_chunks: list[str], user_question: str, previous_conversations: list[str] = None):
    conversation_context = ""
    if previous_conversations:
        conversation_context = f"""
    Previous Conversation:
    {chr(10).join(previous_conversations)}
    """
    
    prompt = f"""
    You are a question-answering assistant.

    Use the context below to answer the question.

    If the answer is not present in the context, say exactly: "I don't know".
    {conversation_context}
    Context:
    {retrieved_chunks}

    Question:
    {user_question}
    """ 
    return prompt

def get_answer_from_llm(prompt: str):

    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )

    return response.output_text
    