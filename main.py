from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np
import tempfile

app = FastAPI()

# ✅ Add CORS middleware (allowing all origins for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://face-recognition-zpwh.onrender.com"],  # You can restrict this to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/liveness")
async def liveness(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            cv2.imwrite(temp.name, img)
            path = temp.name

        # Run DeepFace with anti-spoofing
        results = DeepFace.extract_faces(
            img_path=path,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            anti_spoofing=True
        )

        if not results:
            return JSONResponse(status_code=400, content={"error": "No face detected"})

        face = results[0]
        return JSONResponse(content={
            "is_real": face.get("is_real"),
            "antispoof_score": face.get("antispoof_score"),
            "confidence": face.get("confidence"),
            "eye_distance": face.get("eye_distance")
        })

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
