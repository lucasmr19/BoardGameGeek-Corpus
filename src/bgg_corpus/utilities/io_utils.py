import os
import json
import pandas as pd

def load_json(path):
    """Carga un archivo JSON y devuelve un dict, o None si no existe."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_csv(path):
    """Carga un CSV como DataFrame (o DataFrame vacío si no existe)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)