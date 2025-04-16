from sentence_transformers import SentenceTransformer, util

# Load once at app start
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Store all company names and their embeddings
