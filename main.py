from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import cv2
import numpy as np
import tempfile

app = FastAPI()

@app.post("/liveness")
async def liveness(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image format")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            cv2.imwrite(temp.name, img)
            path = temp.name

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
