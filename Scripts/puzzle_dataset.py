import csv
import pickle
import chess
import numpy as np

from pathlib import Path
from board import encode_board

# region : Variables Globales

DATA_DIR = Path("Data/Lichess Puzzle Database")
PUZZLE_FILE = DATA_DIR / "lichess_db_puzzle.csv"

OUT_DIR = Path("Data/Processed Database/puzzles")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limite le nombre de puzzles pour éviter de surcharger la mémoire et accélérer le processus de développement

MAX_PUZZLES = 500_000

# X: positions encodées
# y: coups légaux (sous forme d'entiers)
# move_to_int: indexation des coups 'sous forme d'entier' vers leur véritable représentation (ex: e2e4)

X = []
y = []

move_to_int = {}

next_id = 0
puzzles_read = 0

# endregion

def get_move_id(uci):

    """
    Attribue un entier unique à chaque coup légal rencontré.
     - uci: coup en notation UCI
    """

    global next_id

    # Si le coup n'a pas encore été rencontré,
    # on lui attribue un nouvel entier unique

    if uci not in move_to_int:
        move_to_int[uci] = next_id
        next_id += 1

    return move_to_int[uci]

# region : Chargement des puzzles

with open(PUZZLE_FILE, encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for i, row in enumerate(reader):

        if i >= MAX_PUZZLES:
            break

        print(f"Reading puzzle {i + 1}", end="\r")

        # On initialise le plateau à partir de la position FEN fournie
        board = chess.Board(row["FEN"])

        # Les coups sont stockés en notation UCI
        moves = row["Moves"].split()

        # 1. On pousse le premier coup adverse
        #
        # La position affichée au joueur est celle
        # après ce premier coup

        board.push_uci(moves[0])

        # 2. On encode ensuite chaque position de la résolution
        #
        # Pour chaque coup :
        #
        # - encode la position actuelle du plateau et l'ajoute à X
        # - encode le coup en entier et l'ajoute à y
        # - met à jour le plateau en jouant le coup
        #
        # De cette manière on crée un dataset de positions
        # & coups tactiques associés

        for uci in moves[1:]:

            move = chess.Move.from_uci(uci)

            X.append(encode_board(board))
            y.append(get_move_id(uci))

            board.push(move)

        puzzles_read += 1

print(f"\nPuzzles: {puzzles_read}")
print(f"Positions: {len(y)}")
print(f"Unique moves: {len(move_to_int)}")

# endregion

# region : Sauvegarde du Dataset

# 1. On convertit les listes Python en tableaux NumPy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# 2. On enregistre les positions
np.save(OUT_DIR / "X_puzzles.npy", X)

# 3. On enregistre les coups joués (encodés en entiers)
np.save(OUT_DIR / "y_puzzles.npy", y)

# 4. On enregistre la correspondance entre les coups encodés
# et leur représentation réelle

with open(OUT_DIR / "move_to_int_puzzles.pkl", "wb") as f:
    pickle.dump(move_to_int, f)

# endregion