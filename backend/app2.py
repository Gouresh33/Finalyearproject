import base64
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from face_module import verify_face_from_image_bytes
from voice_module import verify_voice_from_audio_bytes

app = Flask(__name__)
CORS(app)

SESSION_TIMEOUT = 60  # seconds

SESSION_STORE = {}

FINGERPRINT_DB = {
    1: "Gouresh",
    2: "Ashlesh",
    3: "Mandar"
}

# ---------------- SESSION CLEANUP ----------------
def cleanup_sessions():
    while True:
        now = time.time()
        expired = [
            sid for sid, data in SESSION_STORE.items()
            if now - data["created_at"] > SESSION_TIMEOUT
        ]
        for sid in expired:
            del SESSION_STORE[sid]
        time.sleep(10)

threading.Thread(target=cleanup_sessions, daemon=True).start()

# ---------------- START AUTH ----------------
@app.route("/start-auth", methods=["POST"])
def start_auth():
    data = request.get_json()

    face_b64 = data.get("face_image")
    voice_b64 = data.get("voice_audio")

    if not face_b64 or not voice_b64:
        return jsonify({"status": "error"}), 400

    if "," in face_b64:
        face_b64 = face_b64.split(",")[1]

    face_bytes = base64.b64decode(face_b64)
    voice_bytes = base64.b64decode(voice_b64)

    face_name, face_ok = verify_face_from_image_bytes(face_bytes)
    voice_name, voice_ok = verify_voice_from_audio_bytes(voice_bytes)

    if not face_ok or not voice_ok or face_name != voice_name:
        return jsonify({"status": "denied"}), 403

    session_id = str(time.time())

    SESSION_STORE[session_id] = {
        "identity": face_name,
        "status": "waiting_fp",
        "created_at": time.time()
    }

    return jsonify({
        "status": "face_voice_ok",
        "session_id": session_id
    })

# ---------------- ACTIVE SESSION FOR ESP32 ----------------
@app.route("/active-session")
def active_session():
    for sid, data in SESSION_STORE.items():
        if data["status"] == "waiting_fp":
            return jsonify({"session_id": sid})
    return jsonify({"session_id": None})

# ---------------- FINGER AUTH ----------------
@app.route("/finger-auth", methods=["POST"])
def finger_auth():
    data = request.get_json()
    sid = data.get("session_id")
    finger_id = data.get("fingerprint_id")

    if sid not in SESSION_STORE:
        return jsonify({"status": "invalid"}), 400

    identity = FINGERPRINT_DB.get(int(finger_id))

    if identity == SESSION_STORE[sid]["identity"]:
        SESSION_STORE[sid]["status"] = "granted"
    else:
        SESSION_STORE[sid]["status"] = "denied"

    return jsonify({"status": SESSION_STORE[sid]["status"]})

# ---------------- SESSION STATUS FOR BROWSER ----------------
@app.route("/session-status/<sid>")
def session_status(sid):
    if sid not in SESSION_STORE:
        return jsonify({"status": "expired"})
    return jsonify({"status": SESSION_STORE[sid]["status"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)