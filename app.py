from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import requests
import os
import logging

# -------------------- App Setup --------------------

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY environment variable is not set")

HF_MODEL_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------- Models --------------------

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

# -------------------- Health --------------------

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

# -------------------- HF Call --------------------

def call_hf_embeddings(texts: List[str]):
    logger.info("Calling Hugging Face embeddings API")

    payload = {
        "inputs": texts   # ALWAYS list
    }

    try:
        response = requests.post(
            HF_MODEL_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )
    except requests.RequestException as e:
        logger.error(f"HF request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reach Hugging Face API"
        )

    logger.info(f"HF response status: {response.status_code}")

    if response.status_code == 503:
        raise HTTPException(
            status_code=503,
            detail="Hugging Face model is loading. Retry shortly."
        )

    if response.status_code == 401:
        raise HTTPException(
            status_code=500,
            detail="Invalid Hugging Face API key"
        )

    if response.status_code != 200:
        logger.error(f"HF error {response.status_code}: {response.text}")
        raise HTTPException(
            status_code=500,
            detail=response.text
        )

    return response.json()

# -------------------- Endpoints --------------------

@app.post("/embed")
def embed(req: TextRequest):
    embeddings = call_hf_embeddings([req.text])  # wrap in list
    return {"embedding": embeddings[0]}          # unwrap single vector

@app.post("/embed/batch")
def embed_batch(req: BatchTextRequest):
    embeddings = call_hf_embeddings(req.texts)
    return {"embeddings": embeddings}
