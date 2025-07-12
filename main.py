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
    
# Load once on startup
with open("cleaned_recipes.pkl", "rb") as f:
    recipes = pickle.load(f)

index = faiss.read_index("recipe_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(req: SearchRequest):
    query_vector = model.encode([req.query])
    D, I = index.search(np.array(query_vector).astype("float32"), req.top_k)
    return {"results": [recipes[i] for i in I[0]]}
