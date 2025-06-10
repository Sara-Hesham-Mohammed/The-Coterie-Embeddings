from user_DTO import UserDTO
from user_embedding import create_user_text, get_embedding


def get_users():
    user1 = UserDTO(
        id=1,
        tags=["programming", "gaming"],
        country="USA",
        languages=["English", "Spanish"],
        preferred_group_size="small"
    )

    user2 = UserDTO(
        id=2,
        tags=["art", "music", "reading"],
        country="Canada",
        languages=["English", "French"],
        preferred_group_size="large"
    )

    user3 = UserDTO(
        id=3,
        tags=["sports", "gaming"],
        country="UK",
        languages=["English"],
        preferred_group_size="medium"
    )

    # Additional users
    user4 = UserDTO(
        id=4,
        tags=["programming", "art", "technology"],
        country="Germany",
        languages=["English", "German"],
        preferred_group_size="medium"
    )

    user5 = UserDTO(
        id=5,
        tags=["music", "gaming", "movies"],
        country="Japan",
        languages=["Japanese", "English"],
        preferred_group_size="small"
    )

    user6 = UserDTO(
        id=6,
        tags=["sports", "fitness", "health"],
        country="Australia",
        languages=["English"],
        preferred_group_size="large"
    )

    users = [user1, user2, user3, user4, user5, user6]
    return users

if __name__ == "__main__":
    print("Generating embeddings for users...")
    all_users = get_users()
    for user in all_users:
        print(f"Processing User ID: {user.id}")
        user_dict = user.model_dump()
        usr_txt = create_user_text(user_dict)
        print(f"User ID: {user.id}, Text Representation: {usr_txt}")
        embedding = get_embedding(usr_txt)
        print(f"Embedding for User ID {user.id} generated successfully.\n Embedding shape: {embedding}\n")