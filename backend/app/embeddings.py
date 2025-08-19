from sentence_transformers import SentenceTransformer
from ai_utils import generate_embedding
# Load 384-dim Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = generate_embedding("climate change")  # returns a list of 384 floats
print(emb[:5])
def get_embedding(text: str):
    # Returns a list of floats (length 384)
    return model.encode(text).tolist()
