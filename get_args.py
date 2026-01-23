from omegaconf import OmegaConf
import json
import os

def get_config():
    json_path = os.path.join(os.path.dirname(__file__), "args.json")

    with open(json_path, "r") as f:
        cfg = OmegaConf.create(json.load(f))

    return cfg