# GokulEye

Image-based animal type classification (cattle vs buffalo) with Grad-CAM explanations, skin-condition screening, a livestock assistant, a vet directory and a record history.

## Run
```
pip install -r requirements.txt
python backend/train.py path/to/dataset      # folders: Cattle/, Buffalo/ -> writes models/
cp .env.example .env                          # add LLM_API_KEY for Skin and Assistant tabs
uvicorn backend.app:app --reload
```
Open http://localhost:8000. The classifier returns a clear "model not trained" message until `models/gokuleye.keras` exists; no predictions are ever faked. Accuracy shown in the app comes from `models/metrics.json`, written by the training run on a held-out test split.

## Deploy
One service: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` (Render, Railway, Hugging Face Spaces with Docker). Free tiers sleep when idle; the first request after idle is slow.

## Notes
- Add real clinics to `data/vets.csv` (name,type,address,district,phone,lat,lng,open_24x7,services). Verify numbers before adding.
- Skin screening uses an image-capable language model, not a trained CNN. It is screening support, not a diagnosis.
- National livestock helpline: 1962.
