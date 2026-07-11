"""
Mitha SolarLab — Render-optimized single-server app.

Memory-lean build for Render's free tier (512 MB RAM):
  * no pandas (numpy only)
  * Random Forest loaded from chunked files with mmap (trees stay on disk,
    the OS pages them in on demand instead of holding ~200 MB in RAM)
  * the Ensemble is computed as the exact weighted average of
    XGBoost + LightGBM + RandomForest (identical numbers to the original
    VotingRegressor — same weights, same math)

Run locally:  uvicorn main:app --reload      → http://127.0.0.1:8000
On Render:    uvicorn main:app --host 0.0.0.0 --port $PORT

Default admin  →  admin@mithasolarlab.com  /  MithaSolar@2026
"""
import os, json, glob, sqlite3, hashlib, secrets, datetime, warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

FEATS = ['CellL','ITODop','ZnODop','tETL1','tETL2','tHTL','tAbs','tAg','Dop',
         'OGRScale','LUMO','Eg','eSRVZnO','hSRVZnO','ITOEA','DitMid','Rshunt','Rseries']

# ---------------- load models (memory-lean) ----------------
MF = "model_files"
XGB     = joblib.load(f"{MF}/xgb_model.pkl")
LGB     = joblib.load(f"{MF}/lgb_model.pkl")
SVM     = joblib.load(f"{MF}/svm_model.pkl")
LR      = joblib.load(f"{MF}/lr_model.pkl")
IMPUTER = joblib.load(f"{MF}/imputer.pkl")
SCALER  = joblib.load(f"{MF}/scaler.pkl")
METRICS = json.load(open(f"{MF}/model_metrics.json"))
ENS_W   = np.array(json.load(open(f"{MF}/ensemble_weights.json"))["weights"])

# Random Forest: shell + memory-mapped tree chunks (trees stay on disk)
RF = joblib.load(f"{MF}/rf_shell.pkl")
trees = []
for path in sorted(glob.glob(f"{MF}/rf_trees_*.pkl")):
    trees.extend(joblib.load(path, mmap_mode="r"))
RF.estimators_ = trees

def _prep(row: dict) -> np.ndarray:
    """dict -> imputed numpy row in the exact training feature order."""
    X = np.array([[float(row[f]) for f in FEATS]], dtype=float)
    X = np.where(np.isfinite(X), X, np.nan)
    return IMPUTER.transform(X)

def predict_with(key: str, row: dict) -> float:
    X = _prep(row)
    if key == "xgboost":       return float(XGB.predict(X)[0])
    if key == "lightgbm":      return float(LGB.predict(X)[0])
    if key == "random_forest": return float(RF.predict(X)[0])
    if key == "svm":           return float(SVM.predict(SCALER.transform(X))[0])
    if key == "linear":        return float(LR.predict(SCALER.transform(X))[0])
    if key == "ensemble":      # exact VotingRegressor math: weighted average
        preds = np.array([XGB.predict(X)[0], LGB.predict(X)[0], RF.predict(X)[0]])
        return float(np.average(preds, weights=ENS_W))
    raise HTTPException(400, f"Unknown model '{key}'.")

MODEL_KEYS = ["xgboost","lightgbm","random_forest","svm","linear","ensemble"]

# ---------------- database ----------------
DB = os.environ.get("DB_PATH", "solarlab.db")

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def now(): return datetime.datetime.now().isoformat(timespec="seconds")
def hash_pw(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()

def init_db():
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        contact TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
        model TEXT NOT NULL, params_json TEXT NOT NULL,
        predicted_eff REAL NOT NULL, created_at TEXT NOT NULL);
    """)
    if not c.execute("SELECT 1 FROM users WHERE is_admin=1").fetchone():
        salt = secrets.token_hex(16)
        c.execute("INSERT OR IGNORE INTO users(name,contact,pw_hash,salt,is_admin,created_at) VALUES(?,?,?,?,1,?)",
                  ("Sowmitha (Admin)", "admin@mithasolarlab.com", hash_pw("MithaSolar@2026", salt), salt, now()))
    c.commit(); c.close()

def auth_user(authorization, admin_required=False):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in first.")
    token = authorization.split(" ", 1)[1]
    c = db()
    row = c.execute("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?",
                    (token,)).fetchone()
    c.close()
    if not row: raise HTTPException(401, "Session expired — please sign in again.")
    if admin_required and not row["is_admin"]: raise HTTPException(403, "Admin access only.")
    return row

# ---------------- app ----------------
app = FastAPI(title="Mitha SolarLab API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
init_db()

class RegisterIn(BaseModel):
    name: str; contact: str; password: str
class LoginIn(BaseModel):
    contact: str; password: str
class PredictIn(BaseModel):
    CellL: float; ITODop: float; ZnODop: float; tETL1: float; tETL2: float
    tHTL: float; tAbs: float; tAg: float; Dop: float; OGRScale: float
    LUMO: float; Eg: float; eSRVZnO: float; hSRVZnO: float; ITOEA: float
    DitMid: float; Rshunt: float; Rseries: float
    model: str = "ensemble"

@app.post("/auth/register")
def register(body: RegisterIn):
    contact = body.contact.strip().lower()
    if len(body.password) < 6: raise HTTPException(400, "Password must be at least 6 characters.")
    if not contact or not body.name.strip(): raise HTTPException(400, "Name and email/phone are required.")
    c = db()
    if c.execute("SELECT 1 FROM users WHERE contact=?", (contact,)).fetchone():
        c.close(); raise HTTPException(400, "That email/phone is already registered — try signing in.")
    salt = secrets.token_hex(16)
    c.execute("INSERT INTO users(name,contact,pw_hash,salt,created_at) VALUES(?,?,?,?,?)",
              (body.name.strip(), contact, hash_pw(body.password, salt), salt, now()))
    uid = c.execute("SELECT id FROM users WHERE contact=?", (contact,)).fetchone()["id"]
    token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)", (token, uid, now()))
    c.commit(); c.close()
    return {"token": token, "name": body.name.strip(), "is_admin": False}

@app.post("/auth/login")
def login(body: LoginIn):
    c = db()
    row = c.execute("SELECT * FROM users WHERE contact=?", (body.contact.strip().lower(),)).fetchone()
    if not row or hash_pw(body.password, row["salt"]) != row["pw_hash"]:
        c.close(); raise HTTPException(401, "Wrong email/phone or password.")
    token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO sessions(token,user_id,created_at) VALUES(?,?,?)", (token, row["id"], now()))
    c.commit(); c.close()
    return {"token": token, "name": row["name"], "is_admin": bool(row["is_admin"])}

@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        c = db(); c.execute("DELETE FROM sessions WHERE token=?", (authorization.split(" ",1)[1],)); c.commit(); c.close()
    return {"ok": True}

@app.post("/predict")
def predict(p: PredictIn, authorization: str | None = Header(default=None)):
    user = auth_user(authorization)
    key = p.model.lower()
    if key not in MODEL_KEYS: raise HTTPException(400, f"Unknown model '{p.model}'.")
    row = p.dict(); row.pop("model")
    pred = round(predict_with(key, row), 4)
    c = db()
    c.execute("INSERT INTO predictions(user_id,model,params_json,predicted_eff,created_at) VALUES(?,?,?,?,?)",
              (user["id"], key, json.dumps(row), pred, now()))
    c.commit(); c.close()
    return {"predicted_eff": pred, "model_used": key}

@app.get("/history")
def history(authorization: str | None = Header(default=None)):
    user = auth_user(authorization)
    c = db()
    rows = c.execute("""SELECT id, model, predicted_eff, created_at, params_json
                        FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 100""",
                     (user["id"],)).fetchall()
    c.close()
    return [dict(r) for r in rows]

@app.get("/stats")
def stats():
    c = db()
    tests = c.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]
    users = c.execute("SELECT COUNT(*) n FROM users WHERE is_admin=0").fetchone()["n"]
    c.close()
    return {"total_tests": tests, "total_users": users}

@app.get("/models")
def models(): return METRICS

@app.get("/admin/users")
def admin_users(authorization: str | None = Header(default=None)):
    auth_user(authorization, admin_required=True)
    c = db()
    rows = c.execute("""SELECT u.id, u.name, u.contact, u.created_at,
                               COUNT(p.id) AS tests, MAX(p.created_at) AS last_test
                        FROM users u LEFT JOIN predictions p ON p.user_id=u.id
                        WHERE u.is_admin=0 GROUP BY u.id ORDER BY u.id DESC""").fetchall()
    c.close()
    return [dict(r) for r in rows]

@app.get("/admin/users/{uid}/predictions")
def admin_user_predictions(uid: int, authorization: str | None = Header(default=None)):
    auth_user(authorization, admin_required=True)
    c = db()
    rows = c.execute("""SELECT id, model, predicted_eff, created_at, params_json
                        FROM predictions WHERE user_id=? ORDER BY id DESC""", (uid,)).fetchall()
    c.close()
    return [dict(r) for r in rows]

app.mount("/", StaticFiles(directory="static", html=True), name="site")
