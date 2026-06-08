import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import json


def get_embedding_from_url(image_url: str):
    """
    Download image from Cloudinary URL and return 128-dim face embedding.
    Uses model='small' to ensure consistent 128-dim output.
    """
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_array = np.array(img)
        encodings = face_recognition.face_encodings(img_array, model="small")
        if encodings:
            return encodings[0].tolist()
        print(f"No face detected in image: {image_url}")
        return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def get_embedding_from_frame(frame_rgb):
    """
    Get 128-dim face embeddings from a live BGR→RGB camera frame.
    Uses model='small' to match stored embeddings.
    Returns list of embeddings (one per face detected).
    """
    return face_recognition.face_encodings(frame_rgb, model="small")


def match_face(live_embedding, stored_embedding_json: str, threshold: float = 0.5):
    """
    Compare a live 128-dim embedding against a stored JSON embedding.
    Returns (matched: bool, distance: float).
    Lower distance = better match. Threshold 0.5 is standard.
    """
    try:
        stored = np.array(json.loads(stored_embedding_json))
        live   = np.array(live_embedding)

        if stored.shape != live.shape:
            print(f"Shape mismatch: live={live.shape}, stored={stored.shape}")
            return False, 1.0

        distance = face_recognition.face_distance([stored], live)[0]
        return bool(distance < threshold), float(distance)

    except Exception as e:
        print(f"Match error: {e}")
        return False, 1.0