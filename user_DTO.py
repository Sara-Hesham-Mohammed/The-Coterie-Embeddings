from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any


class UserDTO(BaseModel):
    id: int
    tags: List[str]
    country: str
    languages: List[str]

