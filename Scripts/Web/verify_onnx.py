import chess
import torch
import pickle
import numpy as np
import onnxruntime as ort

from pathlib import Path
from model import ChessCNN
from board import encode_board

# region : Chargement des ressources

MODELS_DIR = Path("Models")
DATA_DIR   = Path("Data/Processed Database")
WEB_DIR    = Path("Web")

with open(DATA_DIR / "move_to_int.pkl", "rb") as f:
    move_to_int = pickle.load(f)

num_moves = len(move_to_int)

model_path = sorted(MODELS_DIR.glob("*.pth"))[-1]
model = ChessCNN(num_moves)

state = torch.load(model_path, map_location="cpu", weights_only=True)

if "model_state" in state:
    model.load_state_dict(state["model_state"])

else:
    model.load_state_dict(state)

model.eval()

# endregion

# region : Vérification ONNX

# Modèle ONNX exporté
ort_session = ort.InferenceSession(str(WEB_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
print("Vérification PyTorch vs ONNX sur 5 positions...\n")

board = chess.Board()
moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]

for i, uci in enumerate(moves):

    encoded = encode_board(board)
    x_np    = encoded[np.newaxis].astype(np.float32)  # (1, 13, 8, 8)
    x_torch = torch.tensor(x_np)

    with torch.no_grad():
        out_torch = model(x_torch).numpy()[0]

    # Inférence ONNX
    out_onnx = ort_session.run(None, {"input": x_np})[0][0]

    # Comparaison
    max_diff = np.abs(out_torch - out_onnx).max()
    top_torch = np.argmax(out_torch)
    top_onnx  = np.argmax(out_onnx)

    status = "V" if top_torch == top_onnx else "X"
    print(f"Position {i+1} | {status} | meilleur coup identique : {top_torch == top_onnx} | diff max : {max_diff:.6f}")

    board.push(chess.Move.from_uci(uci))

# endregion

print("\nVérification terminée.")
print("Si tous les V sont verts, le modèle ONNX est prêt pour le web.")