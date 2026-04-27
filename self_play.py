import torch
import chess
import numpy as np

from board import encode_board
from engine import choose_move_rl

def reward_from_board(board: chess.Board) -> float:
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate():
        return -0.2
    if board.is_check():
        return -0.1
    return 0.0

def play_self_game(model, epsilon, max_moves=300, gamma=0.99):

    board = chess.Board()
    history = []

    while not board.is_game_over() and len(history) < max_moves:

        history.append((encode_board(board), board.turn, board.copy()))
        move = choose_move_rl(board, model, epsilon)
        board.push(move)

    if board.result() == "1-0":
        z = 1.0

    elif board.result() == "0-1":
        z = -1.0

    else:
        z = 0.0

    X, y = [], []

    with torch.no_grad():

        for i in range(len(history) - 1):

            state, turn, board_i = history[i]
            next_state, _, _ = history[i + 1]

            v_next = model(torch.tensor(next_state).unsqueeze(0)).item()

            r = reward_from_board(board_i)
            target = r + gamma * v_next

            target = max(-1.0, min(1.0, target))

            if turn == chess.BLACK:
                target = -target

            X.append(state)
            y.append(target)

        final_state, turn, _ = history[-1]
        final_value = z if turn == chess.WHITE else -z

        X.append(final_state)
        y.append(final_value)

    return X, y

def generate_selfplay_data(model, games, epsilon):

    X_all, y_all = [], []

    for g in range(games):

        X, y = play_self_game(model, epsilon)

        X_all.extend(X)
        y_all.extend(y)

        if (g + 1) % 50 == 0:
            print(f"[Self-play] {g+1}/{games}")

    return np.array(X_all, np.float32), np.array(y_all, np.float32)