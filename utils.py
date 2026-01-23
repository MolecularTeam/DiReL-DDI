import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
