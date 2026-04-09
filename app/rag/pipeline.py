from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Init Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV", "us-east-1-aws")
)

INDEX_NAME = os.getenv("PINECONE_INDEX", "genai-qa")

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the provided context to answer the question accurately.
If the answer is not in the context, say "I don't have enough information to answer this."
Always cite your sources.

Context:
{summaries}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["summaries", "question"]
)


def build_rag_chain():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def answer_question(chain, question: str) -> dict:
    result = chain({"question": question})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]],
    }


# Singleton chain
_chain = None

def get_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain
