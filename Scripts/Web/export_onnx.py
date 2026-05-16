import json
import torch
import pickle

from pathlib import Path
from model import ChessCNN

MODELS_DIR  = Path("Models")
DATA_DIR    = Path("Data/Processed Database")

OUT_DIR     = Path("Web")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# region : Chargement du modèle

with open(DATA_DIR / "move_to_int.pkl", "rb") as f:
    move_to_int = pickle.load(f)

num_moves = len(move_to_int)
print(f"Coups uniques : {num_moves}")

# Recherche du modèle final le plus récent
model_files = sorted(MODELS_DIR.glob("*.pth"))

if not model_files:
    raise FileNotFoundError("Aucun modèle final trouvé dans ../Models/")

model_path = model_files[-1]
print(f"Modèle source : {model_path.name}")

# endregion

# region : Chargement des poids

model = ChessCNN(num_moves)
state = torch.load(model_path, map_location="cpu", weights_only=True)

if "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)

model.eval()

# endregion

# region : Export ONNX

dummy_input = torch.zeros(1, 13, 8, 8)
onnx_path = OUT_DIR / "model.onnx"

torch.onnx.export(model,
                  dummy_input,
                  onnx_path,

                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                  opset_version=17)

print(f"Modèle ONNX exporté : {onnx_path}")
print(f"Taille              : {onnx_path.stat().st_size / 1e6:.2f} MB")

# endregion

# Le navigateur ne peut pas lire les fichiers pickle Python
# On convertit donc move_to_int en JSON pour le charger en JS

json_path = OUT_DIR / "move_to_int.json"

with open(json_path, "w") as f:
    json.dump(move_to_int, f)

print(f"move_to_int exporté : {json_path}")
print(f"Taille              : {json_path.stat().st_size / 1e3:.1f} KB")

print("\nDone. Les fichiers sont prêts dans ../Web/")