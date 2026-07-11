# Mitha SolarLab — AI Powered Solar Cell Efficiency Prediction

Live web app: 6 ML models predict perovskite solar cell efficiency from
18 TCAD device parameters. User accounts, per-user test history, admin panel.
Built by Sowmitha.

## Run locally
```
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

## Deploy on Render (free)
- Build command:  pip install -r requirements.txt
- Start command:  uvicorn main:app --host 0.0.0.0 --port $PORT
- Environment variable:  PYTHON_VERSION = 3.11.9

## Notes
- Memory-optimized build: Random Forest is stored in chunks and memory-mapped;
  the ensemble is the exact weighted average of XGB+LGBM+RF (identical
  predictions to the original VotingRegressor, verified to 0 difference).
- Admin login: admin@mithasolarlab.com / MithaSolar@2026  (change in main.py)
- SQLite DB is ephemeral on Render's free tier: user accounts and history
  reset on redeploys/restarts. Fine for demos.
