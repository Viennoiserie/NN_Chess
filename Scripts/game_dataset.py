import chess
import pickle
import chess.pgn
import numpy as np

from pathlib import Path
from board import encode_board

# region : Variables Globales

DATA_DIR = Path("Data/Lichess Elite Database")
OUT_DIR = Path("Data/Processed Database/games")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limite le nombre de parties pour éviter de surcharger la mémoire et accélérer le processus de développement
#
# Permets aussi de créer des :
#
# - modèles entrainés sur plus ou moins de données
# - des tests rapides avec un plus petit nombre de parties

MAX_GAMES = 50_000

# X: positions encodées, y: coups légaux (sous forme d'entiers)
# move_to_int: indexation des coups 'sous forme d'entier' vers leur véritable représentation (ex: e2e4)

X = []
y = []

move_to_int = {}

next_id = 0
games_read = 0

# endregion

def get_move_id(move):

    """
    Attribue un entier unique à chaque coup légal rencontré dans les parties.
     - move: un objet chess.Move représentant le coup à encoder
    """

    global next_id
    uci = move.uci()

    # On convertit le coup en notation UCI
    # Si le coup n'a pas encore été rencontré, on lui attribue un nouvel entier unique

    if uci not in move_to_int:
        move_to_int[uci] = next_id
        next_id += 1

    return move_to_int[uci]

# region : Chargement des parties

# Recherche de tous les fichiers PNG dans le répertoire de données
for pgn_file in sorted(DATA_DIR.glob("*.pgn")):

    # Ouverture d'un fichier PNG dans le repértoire de données
    with open(pgn_file, encoding="utf-8") as f:

        # Vérification du nombre de partie lues
        while games_read < MAX_GAMES:

            # Lecture d'une partie à la fois
            game = chess.pgn.read_game(f)
            print(f"Reading game {games_read+1} from {pgn_file.name}", end="\r")

            if game is None:
                break

            board = game.board()

            # Pour chaque coup de la partie on :
            #
            # - encode la position actuelle du plateau et l'ajoute à X
            # - encode le coup en entier et l'ajoute à y
            # - met à jour le plateau en jouant le coup
            #
            # De cette manière on crée un dataset de positionts & coups légaux associés 
            # (qui ont été joués dans les parties du dataset)

            for move in game.mainline_moves():

                X.append(encode_board(board))
                y.append(get_move_id(move))

                board.push(move)

            games_read += 1

            if games_read >= MAX_GAMES:
                break

print(f"Games: {games_read}")
print(f"Positions: {len(y)}")
print(f"Unique moves: {len(move_to_int)}")

# endregion

# region : Sauvegarde du Dataset

# 1. On convertit les listes Python en tableaux NumPy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# 2. On enregistre les positions
np.save(OUT_DIR / "X.npy", X)

# 3. On enregistre les coups joués (encodés en entiers)
np.save(OUT_DIR / "y.npy", y)

# 4. On enregistre la correspondance entre les coups encodés et leur représentation réelle
with open(OUT_DIR / "move_to_int.pkl", "wb") as f:
    pickle.dump(move_to_int, f)

# endregion