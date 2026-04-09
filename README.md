# GenAI Q&A Assistant — RAG-Based Knowledge Tool

## What It Does
GenAI Q&A is a Retrieval-Augmented Generation (RAG) system that answers natural language questions by retrieving relevant context from a large internal document corpus. It dynamically pulls the most relevant documents from a Pinecone vector database and feeds them as context into the OpenAI API, delivering accurate grounded answers with source attribution.

## The Problem It Solves
General-purpose LLMs hallucinate when answering domain-specific questions outside their training data. RAG solves this by anchoring responses in real retrievable documents, making answers trustworthy, up-to-date, and traceable to a source.

## Tech Stack
- AI/ML: LangChain, OpenAI API (GPT-4), Pinecone
- Backend: Python, FastAPI
- DevOps: Docker, GitHub Actions

## Key Features
- Retrieval-Augmented Generation pipeline with LangChain
- Pinecone vector database for semantic document search
- Source attribution — every answer cites the documents it used
- Chunking and embedding pipeline for ingesting new documents
- FastAPI interface for easy integration

## Results
- Achieved 87% answer relevance score across 500+ internal documents
- Reduced hallucination rate from 35% to under 8%
- Ingestion pipeline processes 100 new documents in under 3 minutes

## Getting Started
git clone https://github.com/BlastOussey/genai-qa.git
cd genai-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py --docs_path ./docs
uvicorn app.main:app --reload

## Project Structure
genai-qa/
├── app/
│   ├── main.py
│   └── rag/
│       └── pipeline.py
├── ingest.py
└── docs/

## Author
Ousseynou Diop
LinkedIn: https://www.linkedin.com/in/ousseynou-diop-946a1a245/
Portfolio: https://my-resume-umber-rho.vercel.app/
