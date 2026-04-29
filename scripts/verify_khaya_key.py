import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("KHAYA_API_KEY")
r = requests.post(
    "https://translation-api.ghananlp.org/tts/v2/synthesize",
    headers={"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": key},
    json={"text": "Akwaaba", "language": "twi", "format": "wav"},
)
print(r.status_code, r.text[:300])
