from sentence_transformers import SentenceTransformer, util

# Load once at app start
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Store all company names and their embeddings
COMPANY_NAME_INDEX = []  # List of dicts like {'fin_code': '100069', 'name': 'Tata Motors', 'embedding': tensor}
