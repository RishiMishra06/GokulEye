"""GokulEye - image based animal type classification (cattle vs buffalo)."""
import base64, csv, json, math, os, re, sqlite3, time, uuid
from pathlib import Path
import cv2, httpx, numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
DB, UPLOADS = ROOT / "data" / "gokuleye.db", ROOT / "data" / "uploads"
MODEL_PATH, SIZE, MAX_BYTES = ROOT / "models" / "gokuleye.keras", 224, 8 * 1024 * 1024
UPLOADS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="GokulEye")
_model = None


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY, tag TEXT, kind TEXT,
        label TEXT, confidence REAL, risk TEXT, note TEXT, image TEXT, created REAL)""")
    return c


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run: python backend/train.py <dataset_dir>")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def classes():
    p = ROOT / "models" / "classes.json"
    return json.loads(p.read_text()) if p.exists() else ["Buffalo", "Cattle"]


def read_image(data: bytes):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Unsupported file. Upload a JPEG, PNG or WebP image.")
    return img


def save_upload(img) -> str:
    name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(UPLOADS / name), img)
    return name


def gradcam(model, x, idx):
    import tensorflow as tf
    gm = tf.keras.Model(model.inputs, [model.get_layer("out_relu").output, model.output])
    with tf.GradientTape() as t:
        conv, pred = gm(x)
        loss = pred[:, idx]
    w = tf.reduce_mean(t.gradient(loss, conv), axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], -1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()


def add_record(tag, kind, label, conf, risk, note, img):
    with db() as c:
        cur = c.execute("INSERT INTO records(tag,kind,label,confidence,risk,note,image,created) VALUES(?,?,?,?,?,?,?,?)",
                        (tag, kind, label, conf, risk, note, save_upload(img), time.time()))
    return cur.lastrowid


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    img = read_image(await file.read())
    model = load_model()
    rgb = cv2.cvtColor(cv2.resize(img, (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype("float32")
    x = np.expand_dims(rgb / 127.5 - 1.0, 0)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000
    idx = int(np.argmax(probs))
    label = classes()[idx]
    cam = cv2.resize(gradcam(model, x, idx), (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    rid = add_record(f"GE-{label[:3].upper()}-{int(time.time()) % 10000:04d}", "classification",
                     label, float(probs[idx]), "low", "", img)
    return {"id": rid, "label": label, "confidence": float(probs[idx]),
            "probabilities": dict(zip(classes(), map(float, probs))),
            "latency_ms": round(ms, 1), "gradcam": b64(overlay)}


def ask_llm(prompt: str, image: bytes | None = None) -> str:
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise HTTPException(503, "Set LLM_API_KEY in .env to enable this feature.")
    parts = [{"text": prompt}]
    if image:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image).decode()}})
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = httpx.post(url, params={"key": key}, json={"contents": [{"parts": parts}]}, timeout=60)
    if r.status_code != 200:
        raise HTTPException(502, "The assistant service returned an error. Check your API key and model name.")
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


NOTICE = ("You give informational decision support to livestock owners in India. You never replace a "
          "registered veterinarian. For emergencies, tell the user to call the 1962 helpline.")


@app.post("/api/skin")
async def skin(file: UploadFile = File(...), animal: str = Form("Cattle")):
    img = read_image(await file.read())
    text = ask_llm(NOTICE + f" Look at this skin photo of a {animal}. Screen for lumpy skin disease, ringworm, "
                   'warts or mange. Reply with JSON only: {"condition": str, "risk": "low|moderate|urgent", '
                   '"confidence": 0-1, "signs": str, "advice": str}.', cv2.imencode(".jpg", img)[1].tobytes())
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0))
    except Exception:
        raise HTTPException(502, "Could not read the screening result. Try a clearer photo.")
    risk = out.get("risk") if out.get("risk") in ("low", "moderate", "urgent") else "moderate"
    add_record(f"GE-SKN-{int(time.time()) % 10000:04d}", "skin", out.get("condition", "Unclear"),
               float(out.get("confidence", 0)), risk, out.get("signs", ""), img)
    return {**out, "risk": risk}


@app.post("/api/chat")
async def chat(payload: dict):
    msg = str(payload.get("message", ""))[:1000]
    if not msg:
        raise HTTPException(400, "Type a question first.")
    ctx = str(payload.get("context", ""))[:500]
    return {"reply": ask_llm(f"{NOTICE}\nLatest result in the app: {ctx or 'none'}\nQuestion: {msg}")}


@app.get("/api/history")
def history():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM records ORDER BY created DESC LIMIT 200")]


@app.delete("/api/history/{rid}")
def delete(rid: int):
    with db() as c:
        c.execute("DELETE FROM records WHERE id=?", (rid,))
    return {"ok": True}


@app.get("/api/stats")
def stats():
    with db() as c:
        q = lambda s: c.execute(s).fetchone()[0]
        out = {"total": q("SELECT COUNT(*) FROM records"),
               "cattle": q("SELECT COUNT(*) FROM records WHERE label='Cattle'"),
               "buffalo": q("SELECT COUNT(*) FROM records WHERE label='Buffalo'"),
               "skin": q("SELECT COUNT(*) FROM records WHERE kind='skin'"),
               "urgent": q("SELECT COUNT(*) FROM records WHERE risk='urgent'")}
    mp = ROOT / "models" / "metrics.json"
    return {**out, "model": json.loads(mp.read_text()) if mp.exists() else None}


@app.get("/api/vets")
def vets(lat: float | None = None, lng: float | None = None):
    p = ROOT / "data" / "vets.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []
    for r in rows:
        try:
            a, b, c2, d = map(math.radians, (lat, float(r["lat"]), lng, float(r["lng"])))
            h = math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin((d - c2) / 2) ** 2
            r["km"] = round(12742 * math.asin(math.sqrt(h)), 1)
        except Exception:
            r["km"] = None
    return sorted(rows, key=lambda r: (r["km"] is None, r["km"] or 0))


app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
