from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# Create a combined text representation for each user
def create_user_text(user):
    print(f"Creating text for user ID: {user['id']}")
    tags = ", ".join(user['tags'])  # Convert list of tags into a string
    languages = ", ".join(user['languages'])  # Convert list of languages into a string
    return (
        f"User ID: {user['id']}, Interests: {tags}, "
        f"Country: {user['country']}, Languages: {languages}, Preferred Group Size: {user['preferred_group_size']}"
    )



# Function to get embedding from text
def get_embedding(text):
    ohe = OneHotEncoder()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    # Get sentence embeddings
    interest_emb = model.encode("machine learning, travel")
    liked_emb = model.encode("Python course, AI TED talk")

    # One-hot encode languages and country
    lang_country = ohe.transform([["English", "India"]]).toarray()

    # Final embedding
    user_embedding = np.concatenate([interest_emb, liked_emb, lang_country.flatten()])

    return embeddings
