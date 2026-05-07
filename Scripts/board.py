import chess
import numpy as np

PIECE_TO_INDEX = {chess.PAWN:   0,
                  chess.KNIGHT: 1,
                  chess.BISHOP: 2,
                  chess.ROOK:   3,
                  chess.QUEEN:  4,
                  chess.KING:   5}

def encode_board(board: chess.Board) -> np.ndarray:

    """
    Encode le plateau en (13, 8, 8)
    12 pièces + 1 dimension pour les coups légaux
    """

    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue

        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_index = PIECE_TO_INDEX[piece.piece_type] + color_offset

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        tensor[piece_index, row, col] = 1.0

    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor