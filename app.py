from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import os
import logging
from huggingface_hub import InferenceClient

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY environment variable is not set")

# Initialize the lightweight Inference Client. This client handles the routing logic correctly to avoid 404/410 errors
client = InferenceClient(api_key=HF_API_KEY)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

# Endpoint to check the status of the service.
@app.get("/")
def read_root():
    return JSONResponse(
        status_code=200,
        content={
            "status": "UP",
            "service": "embedding-service",
            "message": "Embedding Service is running n healthy. Use /embed endpoint!"
        }
    )

def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        logger.info(f"Requesting embeddings for {len(texts)} inputs via HF Client")
        response = client.feature_extraction(
            texts,
            model=MODEL_ID
        )

        # The response is a numpy-like list. We convert it to a standard Python list.
        # Format: [batch_size, sequence_length, hidden_size]
        result = response.tolist() if hasattr(response, "tolist") else response

        # MEAN POOLING
        # The API returns vectors for every word. Java expects ONE vector per sentence. We average the word vectors here.
        pooled_embeddings = []
        for sentence_matrix in result:
            if isinstance(sentence_matrix[0], list):
                num_tokens = len(sentence_matrix)
                dimension = len(sentence_matrix[0])
                # Calculate the mean across tokens
                mean_vec = [sum(col) / num_tokens for col in zip(*sentence_matrix)]
                pooled_embeddings.append(mean_vec)
            else:
                # If the API already returned a flattened vector
                pooled_embeddings.append(sentence_matrix)
                
        return pooled_embeddings

    except Exception as e:
        logger.error(f"HF Client Error: {str(e)}")
        if "503" in str(e):
            raise HTTPException(status_code=503, detail="Model is loading on HF. Retry in 20s.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
def embed(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    embeddings = get_embeddings([req.text])
    return {"embedding": embeddings[0]}

@app.post("/embed/batch")
def embed_batch(req: BatchTextRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Text list is empty")
    embeddings = get_embeddings(req.texts)
    return {"embeddings": embeddings}
