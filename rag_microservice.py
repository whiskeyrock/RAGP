# rag_microservice.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np

app = FastAPI()

# Load RAG assets on startup
with open("cleaned_recipes.pkl", "rb") as f:
    recipes = pickle.load(f)

index = faiss.read_index("recipe_index.faiss")

class SearchRequest(BaseModel):
    embedding: list[float]
    top_k: int = 5

@app.post("/search")
def search(req: SearchRequest):
    vector = np.array([req.embedding]).astype("float32")
    _, I = index.search(vector, req.top_k)
    return {"results": [recipes[i] for i in I[0]]}