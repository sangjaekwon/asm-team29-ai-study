from fastapi import APIRouter

from agents.agent3.schema import Agent3Request, Agent3Response
from agents.agent3.service import Agent3Service

router = APIRouter()
service = Agent3Service()

@router.post("/", response_model=Agent3Response)
def classify_recipe_type(request: Agent3Request):
    return service.classify(request)

