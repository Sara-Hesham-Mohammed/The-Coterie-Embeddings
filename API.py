import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from user_DTO import UserDTO
from user_embedding import create_user_text, get_embedding
from fastapi import HTTPException
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the python backend for the embeddings."}


@app.post("/get-embedding/")
async def return_embedding(request: dict):  # Accept any dict
    # Ensure the request is a valid JSON object and transform to string so it can be validated with the bbuilt in pydantic model
    req = json.dumps(request)
    # Convert the incoming request to UserDTO for validation
    user_dto = UserDTO.model_validate_json(req)

    if not user_dto:
        print("Conversion failed!")
        raise HTTPException(status_code=422,
                            detail="Invalid user data - missing one of the required fields: id, tags, country, languages")

    user = user_dto.model_dump()
    interest, location_lang = create_user_text(user)
    interest_embedding = get_embedding(interest, 0.7)
    location_lang_embedding = get_embedding(location_lang, 0.3)

    # DB service API (node.js) response
    json_data = {"interest_embedding": interest_embedding.tolist(),
                            "location_lang_embedding": location_lang_embedding.tolist()}
    return json_data