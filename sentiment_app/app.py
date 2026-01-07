from tokenizers import Tokenizer

from src.scripts.settings import Settings
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI
from starlette.responses import JSONResponse
from pydantic import BaseModel
from mangum import Mangum


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


app = FastAPI()
handler = Mangum(app)
tokenizer = Tokenizer.from_file(Settings.onnx_tokenizer_path)
ort_session = ort.InferenceSession(Settings.onnx_embedding_model_path)
ort_classifier = ort.InferenceSession(Settings.onnx_classifier_path)
classes = ["negative", "neutral", "positive"]


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    res_dump = request.model_dump()
    value = res_dump.get("text", None)

    if type(value) is not str or value == "":
        return JSONResponse(
            content={"message": "Invalid or empty string"}, status_code=422
        )

    encoded = tokenizer.encode(value)
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = ort_session.run(None, embedding_inputs)[0]

    classifier_input_name = ort_classifier.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = ort_classifier.run(None, classifier_inputs)[0]

    return PredictResponse(prediction=classes[prediction[0]])