import json
import os

def save_data(save_path, data):
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, "training_results.json"), "w") as f:
        json.dump(data, f)
