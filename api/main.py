from fastapi import FastAPI, File, UploadFile
import shutil
from ai_model.predict import predict_image
from scraper.amazon_scraper import search_amazon

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    with open(f"temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label = predict_image("temp.jpg")
    amazon_results = search_amazon(label)
    
    return {
        "predicted_item": label,
        "results": amazon_results
    }
