# engine.py
import chess
import torch
import random

from board import encode_board

def evaluate(board, model):

    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        v = model(x).item()

    return v if board.turn == chess.WHITE else -v

def negamax(board, model, depth):

    if depth == 0 or board.is_game_over():
        return evaluate(board, model)

    best = -1e9

    for move in board.legal_moves:

        board.push(move)
        value = -negamax(board, model, depth - 1)

        board.pop()
        best = max(best, value)

    return best

def choose_move_rl(board, model, epsilon, depth=2):

    moves = list(board.legal_moves)

    if random.random() < epsilon:
        return random.choice(moves)

    best_move = None
    best_value = -1e9

    for move in moves:

        board.push(move)
        value = -negamax(board, model, depth - 1)
        
        board.pop()

        if value > best_value:
            best_value = value
            best_move = move

    return best_move