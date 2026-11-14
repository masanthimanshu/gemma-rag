from fastapi import APIRouter

from schema import ResponseModel

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=ResponseModel)
def health():
    return {"message": "It's Working!"}
