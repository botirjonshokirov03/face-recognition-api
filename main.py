from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import onnxruntime as ort

app = FastAPI()

# Load ONNX model
model_path = "liveness_model.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Preprocess for MobileNetV2 model

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # Batch dim
    return img

@app.post("/liveness")
async def liveness(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        # Preprocess and predict
        input_tensor = preprocess(img)
        output = session.run(None, {input_name: input_tensor})[0]

        exp_scores = np.exp(output[0])
        probs = exp_scores / np.sum(exp_scores)
        fake_prob, real_prob = probs[0], probs[1]

        result = {
            "result": "real" if real_prob > 0.8 else "fake",
            "confidence": float(real_prob),
            "spoof_score": float(fake_prob)
        }

        return JSONResponse(content=result)

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
