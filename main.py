from user_DTO import UserDTO
from user_embedding import create_user_text, get_embedding


def get_users():
    user1 = UserDTO(
        id=1,
        tags=["programming", "gaming"],
        country="USA",
        languages=["English", "Spanish"]
    )

    user2 = UserDTO(
        id=2,
        tags=["art", "music", "reading"],
        country="Canada",
        languages=["English", "French"]
    )

    user3 = UserDTO(
        id=3,
        tags=["sports", "gaming"],
        country="UK",
        languages=["English"]
    )

    # Additional users
    user4 = UserDTO(
        id=4,
        tags=["programming", "art", "technology"],
        country="Germany",
        languages=["English", "German"]
    )

    user5 = UserDTO(
        id=5,
        tags=["music", "gaming", "movies"],
        country="Japan",
        languages=["Japanese", "English"]
    )

    user6 = UserDTO(
        id=6,
        tags=["sports", "fitness", "health"],
        country="Australia",
        languages=["English"]
    )

    users = [user1, user2, user3, user4, user5, user6]
    return users

if __name__ == "__main__":
    print("Generating embeddings for users...")
    all_users = get_users()
    for user in all_users:
        user = user.model_dump()
        print(f"Processing User ID: {user['id']}")
        interest, location_lang = create_user_text(user)
        print(f"User ID: {user['id']}, Text Representation: {interest} \n Location: {location_lang}")
        embedding = get_embedding(interest, 0.7)
        loc_embedding = get_embedding(location_lang, 0.3)
        print(f"Embedding for User ID {user['id']} generated successfully.\n Embedding shape: {embedding.shape}\n Location Embedding shape: {loc_embedding.shape}\n")