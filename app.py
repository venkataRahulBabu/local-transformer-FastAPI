from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed(req: TextRequest):
    vector = model.encode(req.text).tolist()
    return {"embedding": vector}

@app.post("/embed/batch")
def embed_batch(req: BatchTextRequest):
    vectors = model.encode(req.texts).tolist()
    return {"embeddings": vectors}


# << --------------      Steps to run the application :-    ---------------- >>
# source venv/bin/activate
# uvicorn app:app --host 0.0.0.0 --port 8000