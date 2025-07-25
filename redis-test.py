import json
import redis
import os
import user_embedding as embedding_api

print("Starting Redis subscriber...")

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(redis_url)

# r = redis.Redis(host='localhost', port=6379)

pubsub = r.pubsub()
pubsub.subscribe("user_created")

print("Subscribed to 'user_created' channel. Waiting for messages...")

for message in pubsub.listen():
    if message["type"] == "message":
        user_data = json.loads(message["data"])
        print(f"Received user data: {user_data}")
        try:
            text = embedding_api.create_user_text(user_data)
            embedding = embedding_api.get_embedding(text)
        except Exception as e:
            print(f"Error processing user data: {e}")
            continue