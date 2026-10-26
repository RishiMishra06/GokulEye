"""GokulEye server: TFLite inference + language-model helpers, serves the web app."""
import base64, io, json, os, re, time
from collections import defaultdict, deque
from pathlib import Path
import httpx, numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS, WEB = ROOT / "models", ROOT / "web"
app = FastAPI(title="GokulEye")
_it, _head = None, None
_hits = defaultdict(deque)


def limit(request: Request, n: int):
    """Best-effort per-IP rate limit (per server instance)."""
    fwd = request.headers.get("x-forwarded-for")
    ip = (fwd or (request.client.host if request.client else "x")).split(",")[0].strip()
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= n:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    q.append(now)


def interp():
    global _it, _head
    if _it is None:
        p = MODELS / "gokuleye.tflite"
        if not p.exists() or not (MODELS / "head.json").exists():
            raise HTTPException(503, "Model files are missing on the server.")
        from ai_edge_litert.interpreter import Interpreter
        _it = Interpreter(model_path=str(p))
        _it.allocate_tensors()
        _head = json.loads((MODELS / "head.json").read_text())
    return _it


def decode(b64: str):
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(400, "Bad image data.")
    if len(raw) > 6_000_000:
        raise HTTPException(413, "Image is too large.")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Unsupported image file.")


def cam_weights(W, P, c):
    """Exact Grad-CAM channel weights for a global-average-pool + dense + softmax head (up to a positive scale)."""
    return W[:, c] - W @ P


@app.post("/api/classify")
def classify(payload: dict, request: Request):
    limit(request, 60)
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
    cam = np.maximum(A.reshape(-1, A.shape[-1]) @ cam_weights(W, P, c), 0)
    return {"p": P.tolist(), "c": c, "heat": (cam / (cam.max() + 1e-8)).tolist()}


NOTICE = ("You give informational decision support to livestock owners in India. You never replace a "
          "registered veterinarian. For emergencies, tell the user to call the 1962 helpline. Keep answers short.")


def ask_llm(prompt: str, image_b64: str | None = None) -> str:
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise HTTPException(503, "LLM_API_KEY is not set on the server.")
    parts = [{"text": prompt}]
    if image_b64:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('LLM_MODEL', 'gemini-3.5-flash')}:generateContent"
    r = httpx.post(url, headers={"x-goog-api-key": key}, json={"contents": [{"parts": parts}]}, timeout=60)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text
        raise HTTPException(502, f"Language model error {r.status_code}: {msg[:200]}")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


@app.post("/api/chat")
def chat(payload: dict, request: Request):
    limit(request, 12)
    msg = str(payload.get("message", "")).strip()[:1000]
    if not msg:
        raise HTTPException(400, "Type a question first.")
    lang = " Reply in simple Hindi (Devanagari script)." if payload.get("lang") == "hi" else ""
    return {"reply": ask_llm(f"{NOTICE}{lang}\nLatest result in the app: {str(payload.get('context', ''))[:500] or 'none'}\nQuestion: {msg}")}


@app.post("/api/skin")
def skin(payload: dict, request: Request):
    limit(request, 12)
    image = str(payload.get("image", ""))
    decode(image)
    lang = " Write the signs and advice fields in simple Hindi (Devanagari script)." if payload.get("lang") == "hi" else ""
    text = ask_llm(NOTICE + lang + f" Look at this skin photo of a {str(payload.get('animal', 'Cattle'))[:20]}. Screen for lumpy skin disease, "
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
