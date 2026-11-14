import os
import secrets


SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
DATABASE_PATH = "database/drug.db"

MODEL_PATH = "model/model.h5"
TOKENIZER_PATH = "model/tokenizer.pkl"
SCALER_PATH = "model/scaler.pkl"