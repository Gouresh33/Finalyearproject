import base64
import requests 
with open("voice.wav", "rb") as f:
    voice_b64 = base64.b64encode(f.read()).decode()

payload = {
    "face_image": base64.b64encode(b"dummy").decode(),
    "voice_audio": voice_b64
}

r = requests.post("http://127.0.0.1:5000/authenticate", json=payload)
print(r.text)
