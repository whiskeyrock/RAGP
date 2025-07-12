from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "🟢 RAG microservice is running"}
    

