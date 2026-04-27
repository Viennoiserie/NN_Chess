import chess
import torch
import numpy as np
import tkinter as tk
from pathlib import Path

from model import ChessCNN
from board import encode_board
from engine import choose_move_rl

# CONFIG 

MODEL_DIR = Path.home() / "Downloads" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIZE = 64
BOARD_COLOR_LIGHT = "#f0d9b5"
BOARD_COLOR_DARK = "#b58863"
HIGHLIGHT_COLOR = "#aaffaa"

PIECE_UNICODE = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
                 "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}

# UI CLASS

class ChessUI:

    def __init__(self, model, epsilon=0.0):

        self.model = model
        self.epsilon = epsilon

        self.board = chess.Board()
        self.history = []
        self.selected_square = None

        self.root = tk.Tk()
        self.root.title("RL Chess – Human vs Model")

        self.canvas = tk.Canvas(self.root,
                                width=8 * SQUARE_SIZE,
                                height=8 * SQUARE_SIZE)
        
        self.canvas.pack()

        self.status = tk.Label(self.root, text="Your move", font=("Arial", 12))
        self.status.pack()

        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()
        self.draw_pieces()

    # DRAWING 

    def draw_board(self):
        self.canvas.delete("square")

        for row in range(8):
            for col in range(8):

                color = BOARD_COLOR_LIGHT if (row + col) % 2 == 0 else BOARD_COLOR_DARK

                x1 = col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE

                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE

                self.canvas.create_rectangle(x1, y1, x2, y2,
                                            fill=color,
                                            tags="square")

    def draw_pieces(self):
        self.canvas.delete("piece")

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)

            if piece:

                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)

                x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                y = row * SQUARE_SIZE + SQUARE_SIZE // 2

                self.canvas.create_text(x, y,
                                        text=PIECE_UNICODE[piece.symbol()],
                                        font=("Arial", 36),
                                        tags="piece")

    # INPUT 

    def on_click(self, event):
        if self.board.is_game_over():
            return

        col = event.x // SQUARE_SIZE
        row = event.y // SQUARE_SIZE
        square = chess.square(col, 7 - row)

        if self.selected_square is None:

            if self.board.piece_at(square) and self.board.turn == chess.WHITE:
                self.selected_square = square
                self.highlight(square)
        else:
            move = chess.Move(self.selected_square, square)
            self.clear_highlight()
            self.selected_square = None

            if move in self.board.legal_moves:
                self.play_human_move(move)

    def highlight(self, square):

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE

        self.canvas.create_rectangle(x1, y1, x2, y2,
                                     outline="green",
                                     width=4,
                                     tags="highlight")

    def clear_highlight(self):
        self.canvas.delete("highlight")

    # GAME LOGIC 

    def play_human_move(self, move):

        self.history.append((encode_board(self.board), self.board.turn))
        self.board.push(move)
        self.update_ui()

        if not self.board.is_game_over():
            self.root.after(300, self.play_model_move)

    def play_model_move(self):

        self.history.append((encode_board(self.board), self.board.turn))
        move = choose_move_rl(self.board, self.model, self.epsilon)

        self.board.push(move)
        self.update_ui()

    def update_ui(self):

        self.draw_board()
        self.draw_pieces()

        if self.board.is_game_over():
            self.finish_game()

        else:
            self.status.config(text="Your move" if self.board.turn == chess.WHITE else "Model thinking...")

    # END GAME

    def finish_game(self):

        result = self.board.result()
        self.status.config(text=f"Game over: {result}")

        if result == "1-0":
            z = 1.0

        elif result == "0-1":
            z = -1.0

        else:
            z = 0.0

        X, y = [], []

        with torch.no_grad():

            for i in range(len(self.history) - 1):

                state, turn = self.history[i]
                next_state, _ = self.history[i + 1]

                v_next = self.model(torch.tensor(next_state).unsqueeze(0)).item()
                target = 0.99 * v_next

                if turn == chess.BLACK:
                    target = -target

                X.append(state)
                y.append(target)

            final_state, turn = self.history[-1]
            final_value = z if turn == chess.WHITE else -z

            X.append(final_state)
            y.append(final_value)

        np.savez(MODEL_DIR / "human_game.npz",
                 X=np.array(X, np.float32),
                 y=np.array(y, np.float32))

        print("Saved human_game.npz")

# MAIN

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessCNN().to(device)

    latest = MODEL_DIR / "latest.pth"
    
    if latest.exists():
        model.load_state_dict(torch.load(latest, map_location=device))

    model.eval()

    ui = ChessUI(model)
    ui.root.mainloop()