from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV", "us-east-1-aws")
)

INDEX_NAME = os.getenv("PINECONE_INDEX", "genai-qa")


def load_documents(docs_path: str):
    loaders = [
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} documents from {docs_path}")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest(docs_path: str):
    docs = load_documents(docs_path)
    chunks = chunk_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create or update Pinecone index
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=1536, metric="cosine")
        print(f"Created Pinecone index: {INDEX_NAME}")

    Pinecone.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print(f"Ingested {len(chunks)} chunks into Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_path", default="./docs", help="Path to documents folder")
    args = parser.parse_args()
    ingest(args.docs_path)
