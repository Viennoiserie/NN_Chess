import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from model import ChessCNN
from self_play import generate_selfplay_data

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path.home() / "Downloads" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPLAY_SIZE = 200_000
SELFPLAY_GAMES = 100
BATCH_SIZE = 128
ITERATIONS = 100
EPOCHS = 5
LR = 1e-4

EPS_MIN = 0.02
EPS_START = 0.8
EPS_DECAY = 0.98

HUMAN_OVERSAMPLE = 5

def main():

    model = ChessCNN().to(DEVICE)
    latest = MODEL_DIR / "latest.pth"

    if latest.exists():
        model.load_state_dict(torch.load(latest, map_location=DEVICE))
        print("[INFO] Model loaded")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    replay_X, replay_y = [], []
    human_X, human_y = [], []

    human_file = MODEL_DIR / "human_game.npz"

    if human_file.exists():

        data = np.load(human_file)

        human_X.extend(data["X"])
        human_y.extend(data["y"])

        print(f"[INFO] Loaded {len(human_X)} human samples")

    epsilon = EPS_START

    for iteration in range(1, ITERATIONS + 1):

        print(f"\n=== ITERATION {iteration} | epsilon={epsilon:.3f} ===")
        model.eval()

        X_sp, y_sp = generate_selfplay_data(model,
                                            games=SELFPLAY_GAMES,
                                            epsilon=epsilon)

        replay_X.extend(X_sp)
        replay_y.extend(y_sp)

        replay_X = replay_X[-REPLAY_SIZE:]
        replay_y = replay_y[-REPLAY_SIZE:]

        combined_X = []
        combined_y = []

        if human_X:
            combined_X.extend(human_X * HUMAN_OVERSAMPLE)
            combined_y.extend(human_y * HUMAN_OVERSAMPLE)

        combined_X.extend(replay_X)
        combined_y.extend(replay_y)

        X = torch.from_numpy(np.stack(combined_X)).float().to(DEVICE)
        y = torch.from_numpy(np.array(combined_y)).float().unsqueeze(1).to(DEVICE)

        # TRAIN
        model.train()

        for epoch in range(EPOCHS):

            perm = torch.randperm(X.size(0), device=DEVICE)
            total_loss = 0.0

            for i in range(0, X.size(0), BATCH_SIZE):

                idx = perm[i:i + BATCH_SIZE]

                preds = model(X[idx])
                loss = criterion(preds, y[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss:.4f}")

        torch.save(model.state_dict(), MODEL_DIR / f"iter_{iteration:04d}.pth")
        torch.save(model.state_dict(), MODEL_DIR / "latest.pth")
        print("[INFO] Model saved")

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

if __name__ == "__main__":
    main()