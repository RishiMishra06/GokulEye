"""GokulEye server: TFLite inference + Gemini helpers, serves the web app."""
import base64, io, json, os, re
from pathlib import Path
import httpx, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
MODELS, WEB = ROOT / "models", ROOT / "web"
app = FastAPI(title="GokulEye")
_it, _head = None, None


def interp():
    global _it, _head
    if _it is None:
        p = MODELS / "gokuleye.tflite"
        if not p.exists() or not (MODELS / "head.json").exists():
            raise HTTPException(503, "Model files are missing on the server.")
        _it = Interpreter(model_path=str(p))
        _it.allocate_tensors()
        _head = json.loads((MODELS / "head.json").read_text())
    return _it


def decode(b64: str):
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Bad image data.")
    if len(raw) > 6_000_000:
        raise HTTPException(413, "Image is too large.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unsupported image file.")


@app.post("/api/classify")
def classify(payload: dict):
    img = decode(str(payload.get("image", "")))
    x = np.asarray(img.resize((224, 224), Image.BILINEAR), dtype="float32") / 127.5 - 1.0
    it = interp()
    it.set_tensor(it.get_input_details()[0]["index"], x[None])
    it.invoke()
    outs = [it.get_tensor(o["index"]) for o in it.get_output_details()]
    A = next(o for o in outs if o.ndim == 4)[0]
    P = next(o for o in outs if o.ndim == 2)[0]
    c = int(P.argmax())
    W = np.array(_head["W"])
    cam = np.maximum(A.reshape(-1, A.shape[-1]) @ (W[:, c] - W[:, 1 - c]), 0)  # exact Grad-CAM for GAP + dense head
    return {"p": P.tolist(), "c": c, "heat": (cam / (cam.max() + 1e-8)).tolist()}


NOTICE = ("You give informational decision support to livestock owners in India. You never replace a "
          "registered veterinarian. For emergencies, tell the user to call the 1962 helpline. Keep answers short.")


def gemini(prompt: str, image_b64: str | None = None) -> str:
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise HTTPException(503, "LLM_API_KEY is not set on the server.")
    parts = [{"text": prompt}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('LLM_MODEL', 'gemini-2.0-flash')}:generateContent"
    r = httpx.post(url, headers={"x-goog-api-key": key}, json={"contents": [{"parts": parts}]}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, "The assistant service returned an error.")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/api/chat")
def chat(payload: dict):
    msg = str(payload.get("message", "")).strip()[:1000]
    if not msg:
        raise HTTPException(400, "Type a question first.")
    return {"reply": gemini(f"{NOTICE}\nLatest result in the app: {str(payload.get('context', ''))[:500] or 'none'}\nQuestion: {msg}")}


@app.post("/api/skin")
def skin(payload: dict):
    image = str(payload.get("image", ""))
    decode(image)
    text = gemini(NOTICE + f" Look at this skin photo of a {str(payload.get('animal', 'Cattle'))[:20]}. Screen for lumpy skin disease, "
                  'ringworm, warts or mange. Reply with JSON only: {"condition": str, "risk": "low|moderate|urgent", '
                  '"confidence": 0-1, "signs": str, "advice": str}.', image)
    try:
        out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        raise HTTPException(502, "Could not read the screening result. Try a clearer photo.")
    if out.get("risk") not in ("low", "moderate", "urgent"):
        out["risk"] = "moderate"
    return out


if MODELS.exists():
    app.mount("/model", StaticFiles(directory=MODELS), name="model")
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
