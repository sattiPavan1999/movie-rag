from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import json
import shutil, os
import dotenv
dotenv.load_dotenv()

# ---------- CONFIG ----------
MOVIES_JSON_PATH = "data/movies.json"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "movies"
# ----------------------------

def load_movies():
    with open(MOVIES_JSON_PATH, 'r') as file:
        data = json.load(file)

    return data

def movies_to_documents(movies):
    documents = []

    for movie in movies:
        content = f"""
        Movie: {movie['title']}
        Year: {movie['year']}
        Director: {movie['director']}
        Cast: {', '.join(movie['cast'])}
        """

        doc = Document(
            page_content=content.strip(),
            metadata={
                "movie_id": movie["id"],
                "title": movie["title"],
                "year": movie["year"]
            }
        )
        documents.append(doc)

    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def ingest():
    movies = load_movies()
    documents = movies_to_documents(movies)
    chunks = chunk_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print(f"🗑️ Deleted Existing {CHROMA_PERSIST_DIR}")

    # Initialize vector store
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    print(f"✅ Created ChromaDB at {CHROMA_PERSIST_DIR}")

    # Add fresh documents
    vector_store.add_documents(chunks)

    print(f"✅ Ingested {len(chunks)} chunks into ChromaDB")