import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
from tempfile import NamedTemporaryFile

import os

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.services.pipeline import InferencePipeline

app = FastAPI(title="电商评论多模态情感分析与可信度挖掘 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SENTIMENT_CKPT = "outputs/sentiment/best_sentiment.pt"
CREDIBILITY_CKPT = "outputs/credibility/best_credibility.pt"

pipeline = None


@app.on_event("startup")
def startup_event() -> None:
    global pipeline
    credibility_path = CREDIBILITY_CKPT if Path(CREDIBILITY_CKPT).exists() else None
    if Path(SENTIMENT_CKPT).exists():
        pipeline = InferencePipeline(
            sentiment_ckpt=SENTIMENT_CKPT,
            credibility_ckpt=credibility_path,
        )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "sentiment_model_loaded": pipeline is not None,
        "credibility_model_loaded": pipeline is not None and pipeline.credibility_model is not None,
    }


@app.post("/predict")
async def predict_review(
    text: str = Form(...),
    rating: float = Form(5.0),
    timestamp: str = Form("2026-01-01 00:00:00"),
    user_review_count: float = Form(0.0),
    user_account_days: float = Form(1.0),
    helpful_votes: float = Form(0.0),
    verified_purchase: float = Form(0.0),
    image: UploadFile = File(...),
):
    if pipeline is None:
        return {"code": 500, "message": "模型尚未加载，请先训练并放置好 checkpoint 文件。"}

    suffix = Path(image.filename).suffix or ".jpg"
    tmp_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await image.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name

        result = pipeline.predict(
            text=text,
            image_path=tmp_path,
            rating=rating,
            timestamp=timestamp,
            user_review_count=user_review_count,
            user_account_days=user_account_days,
            helpful_votes=helpful_votes,
            verified_purchase=verified_purchase,
        )
        return {"code": 200, "message": "success", "data": result}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass