# MediMind: AI patient risk and hospital load platform

Estimates diabetes and heart-disease risk with explanations, forecasts hospital bed
occupancy, shows a model comparison report, and includes an optional Gemini-powered
health Q&A assistant. Deploys to Vercel as a static page plus one Python function.

> Educational decision-support only. Not a medical diagnosis. The bundled models
> are trained on synthetic demo data; use public datasets (e.g. Kaggle Pima Diabetes,
> UCI Heart Disease) for your report. Never use real patient data.

## Run on your PC

Install Python 3.10+ and Git first. Then in a terminal:

```bash
cd medimind
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements-dev.txt
python ml/train.py          # optional: retrain (trained models are already included)
pytest -q                   # run tests
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 (API docs: http://127.0.0.1:8000/docs).

Optional AI assistant: copy `.env.example` to `.env` and paste your free key from
https://aistudio.google.com. The rest of the app works without it.

## Push to GitHub

1. Create an empty repository on github.com (no README), e.g. `medimind`.
2. In the project folder:

```bash
git init
git add .
git commit -m "Initial commit: MediMind"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/medimind.git
git push -u origin main
```

If asked to log in, use a GitHub personal access token as the password
(GitHub > Settings > Developer settings > Personal access tokens).

## Deploy to Vercel (free)

Option A, dashboard: sign in at vercel.com with GitHub, click **Add New > Project**,
import the `medimind` repo, leave all build settings empty, and click **Deploy**.
Add `GEMINI_API_KEY` under Settings > Environment Variables, then redeploy.

Option B, CLI:

```bash
npm i -g vercel
vercel login
vercel            # preview deploy
vercel --prod     # production deploy
vercel env add GEMINI_API_KEY
```

Every `git push` to `main` then redeploys automatically.

## Project layout

```
api/index.py        Vercel entry point (exposes the FastAPI app)
app/                FastAPI code: model inference, forecasting, assistant
ml/train.py         Trains models, exports JSON parameters + metrics
models/             Trained model JSON (committed so Vercel needs no training)
public/index.html   Web interface
tests/              pytest suite
vercel.json         Routing: /api/* goes to the Python function
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /api/schema | Input fields per condition |
| POST | /api/predict | `{"disease":"diabetes","values":{...}}` returns risk and factors |
| GET/POST | /api/forecast/beds | Bed forecast (demo data or your own counts) |
| GET | /api/metrics | Model comparison metrics |
| POST | /api/assistant | `{"question":"..."}` |

## Ideas to extend for your final-year report

Use real public datasets, add a chest X-ray CNN with Grad-CAM (needs a separate
GPU/host, since Vercel functions are too small for PyTorch), add login and a database
(Neon/Supabase free tier), and add a fairness audit by age and sex.
