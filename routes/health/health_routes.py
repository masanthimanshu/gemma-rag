from fastapi import APIRouter


router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
@router.get("/")
def health():
    return {"message": "It's Working!"}
