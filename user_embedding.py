from sentence_transformers import SentenceTransformer

# Create a combined text representation for each user
def create_user_text(user_obj):
    """
    Create a text representation of user interests and location/languages.
    Args:
        user_obj (UserDTO): Custom Pydantic class object containing user data with keys 'id', 'tags', 'country', and 'languages'.

    Returns:
        List[str]: A list containing two strings:
            - Interests: A comma-separated string of user interests.
            - Location and Languages: A string combining country and languages spoken.
    """
    # Convert inputs to strings if they're lists
    tags = ''
    languages = ''
    if isinstance(user_obj['tags'], list):
        tags = ", ".join(user_obj['tags'])
    if isinstance(user_obj['languages'], list):
        languages = ", ".join(user_obj['languages'])

    # Create text representations
    interests_text = f"Interests: {tags}"
    location_text = f"Country: {user_obj['country']}, Languages: {languages}" # combined because they are related and of equivalent importance

    return [interests_text, location_text]


def get_embedding(text , weight):
    """
    Get embeddings by combining different user data components.
    Args:
        text (str): Text representations of user interests or {location and languages they speak}
        weight (float): Weight to apply to the embedding for prioritization

    Returns:
        embedding (np.ndarray): User embedding vector array
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get individual embedding and multiply by a weight to ensure priority to certain features
    single_embedding = model.encode(text) * weight

    return single_embedding


