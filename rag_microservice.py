from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load once on startup
try:
    with open("cleaned_recipes.pkl", "rb") as f:
        recipes = pickle.load(f)

    index = faiss.read_index("recipe_index.faiss")
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("🔥 Failed to start FastAPI app:", e)
    raise e