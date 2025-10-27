from fastapi import FastAPI, HTTPException, Query
from pydantic import HttpUrl
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
import requests
import uvicorn


app = FastAPI(title="Muffin vs Chihuahua Classifier")


session = ort.InferenceSession(
    "bin_class_model.onnx",
    providers=["CPUExecutionProvider"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Use GET /predict?image_url=..."}


@app.get("/predict")
def predict(image_url: HttpUrl = Query(...)):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
        }
        resp = requests.get(str(image_url), timeout=15, headers=headers)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((32, 32))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preprocess image: {e}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: arr})[0]
    logit = float(logits[0][0])
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = "chihuahua" if prob > 0.5 else "muffin"

    return {"label": label, "prob_chihuahua": prob, "prob_muffin": 1.0 - prob}


asgi_app = app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


