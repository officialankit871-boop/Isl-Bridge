import json
import numpy as np
import requests

LANDMARK_PATH = "data/processed/landmarks/000000.npy"
API_URL = "http://127.0.0.1:8000/predict"

# Load real extracted landmarks
landmarks = np.load(LANDMARK_PATH).astype(float)

print("Loaded landmarks:", landmarks.shape)

payload = {
    "landmarks": landmarks.tolist()
}

response = requests.post(
    API_URL,
    json=payload,
    timeout=60,
)

print("\nHTTP Status:", response.status_code)
print("Response:")

try:
    print(json.dumps(response.json(), indent=2))
except Exception:
    print(response.text)