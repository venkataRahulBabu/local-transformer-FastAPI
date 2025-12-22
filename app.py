from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return JSONResponse(
        status_code=200,
        content={
            "status": "UP",
            "service": "hf-transformer",
            "message": "Service is healthy"
        }
    )


def call_hf_embeddings(inputs):
    response = requests.post(
        HF_MODEL_URL,
        headers=HEADERS,
        json={"inputs": inputs},
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"HuggingFace API error: {response.text}"
        )

    return response.json()


@app.post("/embed")
def embed(req: TextRequest):
    embedding = call_hf_embeddings(req.text)
    return {"embedding": embedding}


@app.post("/embed/batch")
def embed_batch(req: BatchTextRequest):
    embeddings = call_hf_embeddings(req.texts)
    return {"embeddings": embeddings}
