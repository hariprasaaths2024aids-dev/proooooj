from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import List
import tempfile
import requests
import os

from embedding import load_document, create_vectorstore
from decision import evaluate_with_llm

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def run_query(payload: QueryRequest):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            response = requests.get(payload.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        docs = load_document(tmp_path)
        vectorstore = create_vectorstore(docs)

        results = []
        for q in payload.questions:
            try:
                raw_answer = evaluate_with_llm(q, vectorstore)
                flat = raw_answer.get("justification", "No justification provided.")
                results.append(flat)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return {"answers": results}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def debug():
    return {
        "env_team_token_set": bool(os.getenv("team_token")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "team_token_value": os.getenv("team_token"),
        "groq_key_prefix": os.getenv("GROQ_API_KEY", "")[:10] + "...",
        "status": "ok"
    }