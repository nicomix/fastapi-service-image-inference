from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.post("/single-inference")
async def singleInference(file: UploadFile):
    return {"filename": file.filename}