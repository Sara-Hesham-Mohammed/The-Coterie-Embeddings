import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from user_DTO import UserDTO
from user_embedding import create_user_text, get_embedding

app = FastAPI()
#to be able to send requests using html file?
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
async def root():
    return {"message": "This is the python backend for the embeddings and clustering(group formation)."}

#### EMBEDDING ENDPOINT ####
@app.post("/get-embedding/")
async def store_embedding(user: UserDTO):
    # serialise the user pydantic object to a dictionary
    user_dict = user.model_dump()
    usr_text = create_user_text(user_dict)
    embedding = get_embedding(usr_text)
    #this is what will get sent to the database service API (node.js)
    json_data = json.dumps({"embedding": embedding.tolist()})
    return json_data
