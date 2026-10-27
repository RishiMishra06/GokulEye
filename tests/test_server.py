import base64, io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from backend import server

client = TestClient(server.app)


def png_b64():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_rejects_bad_image():
    assert client.post("/api/classify", json={"image": "not-an-image"}).status_code == 400


def test_classify_valid_image():
    r = client.post("/api/classify", json={"image": png_b64()})
    assert r.status_code in (200, 503)  # 503 only if model files are absent
    if r.status_code == 200:
        body = r.json()
        assert abs(sum(body["p"]) - 1) < 1e-3 and len(body["heat"]) == 49


def test_chat_needs_message():
    assert client.post("/api/chat", json={"message": ""}).status_code == 400


def test_chat_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert client.post("/api/chat", json={"message": "hello"}).status_code == 503


def test_gradcam_weights_match_numeric_gradient():
    rng = np.random.default_rng(0)
    W, a, c = rng.normal(size=(6, 3)), rng.normal(size=6), 1

    def prob(v):
        z = v @ W
        e = np.exp(z - z.max())
        return (e / e.sum())[c]

    z = a @ W
    P = np.exp(z - z.max()); P /= P.sum()
    g = np.array([(prob(a + 1e-6 * np.eye(6)[i]) - prob(a - 1e-6 * np.eye(6)[i])) / 2e-6 for i in range(6)])
    w = server.cam_weights(W, P, c)
    assert np.dot(g, w) / (np.linalg.norm(g) * np.linalg.norm(w)) > 0.9999


def test_skin_rejects_other_animals(monkeypatch):
    monkeypatch.setattr(server, "ask_llm", lambda *a, **k: '{"valid": false}')
    r = client.post("/api/skin", json={"image": png_b64(), "animal": "Cattle"})
    assert r.status_code == 422


def test_skin_accepts_cattle(monkeypatch):
    monkeypatch.setattr(server, "ask_llm", lambda *a, **k: '{"valid": true, "condition": "Healthy", "risk": "low", "confidence": 0.9, "signs": "-", "advice": "-"}')
    r = client.post("/api/skin", json={"image": png_b64(), "animal": "Cattle"})
    assert r.status_code == 200 and r.json()["risk"] == "low"
