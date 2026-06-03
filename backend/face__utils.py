import face_recognition
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import json

def get_embedding_from_url(image_url: str):
    """Download image from URL and return face embedding."""
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_array = np.array(img)
        encodings = face_recognition.face_encodings(img_array)
        if encodings:
            return encodings[0].tolist()
        return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def get_embedding_from_frame(frame_rgb):
    """Get face embeddings from a live camera frame."""
    encodings = face_recognition.face_encodings(frame_rgb)
    return encodings  # returns list of all faces in frame

def match_face(live_embedding, stored_embedding_json: str, threshold=0.5):
    """Compare live embedding against stored JSON embedding."""
    try:
        stored = np.array(json.loads(stored_embedding_json))
        live = np.array(live_embedding)
        distance = face_recognition.face_distance([stored], live)[0]
        return distance < threshold, float(distance)
    except Exception as e:
        print(f"Match error: {e}")
        return False, 1.0