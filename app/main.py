from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag.pipeline import get_chain, answer_question

app = FastAPI(title="GenAI Q&A API", version="1.0.0")


class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        chain = get_chain()
        result = answer_question(chain, req.question)
        return AnswerResponse(
            question=req.question,
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
