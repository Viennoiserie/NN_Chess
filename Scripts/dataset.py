import chess
import pickle
import chess.pgn
import numpy as np

from pathlib import Path
from board import encode_board

# region : Variables Globales

DATA_DIR = Path("Data/Lichess Elite Database")
OUT_DIR = Path("Data/Processed Database")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limite le nombre de parties pour éviter de surcharger la mémoire et accélérer le processus de développement
#
# Permets aussi de créer des :
#
# - modèles entrainés sur plus ou moins de données
# - des tests rapides avec un plus petit nombre de parties

MAX_GAMES = 50_000
MAX_POSITIONS = MAX_GAMES * 100  

# Sert à pré-allouer la mémoire pour le fichier memmap 
BOARD_SHAPE = encode_board(chess.Board()).shape

# X: positions encodées, y: coups légaux (sous forme d'entiers)
# move_to_int: indexation des coups 'sous forme d'entier' vers leur véritable représentation (ex: e2e4)

# Particularité de X: on utilise un fichier memmap pour éviter de surcharger la RAM
X_PATH = OUT_DIR / "X.dat" 
X_mmap = np.memmap(X_PATH, dtype=np.float32,
                           mode="w+",
                           shape=(MAX_POSITIONS, *BOARD_SHAPE))

y = []
move_to_int = {}

next_id = 0
games_read = 0
position_count = 0

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

                X_mmap[position_count] = encode_board(board)
                y.append(get_move_id(move))

                position_count += 1
                board.push(move)

            games_read += 1
            
            if position_count >= MAX_POSITIONS:
                break

            if games_read >= MAX_GAMES:
                break

print(f"Games: {games_read}")
print(f"Positions: {len(y)}")
print(f"Unique moves: {len(move_to_int)}")

# region : Sauvegarde du Dataset

X_mmap.flush()  

# On s'assure de supprimer les lignes allouées en surplus
# Car en toute logique position_count < MAX_POSITIONS

X_final = np.memmap(X_PATH, dtype=np.float32,
                            mode="r",
                            shape=(position_count, *BOARD_SHAPE))

# 1. On enregistre les positions
np.save(OUT_DIR / "X.npy", np.array(X_final, dtype=np.float32)) 

# On libère la mémoire utilisée par les fichiers memmap
del X_mmap
del X_final
X_PATH.unlink()

# 2. On enregistre les coups joués (encodés en entiers)
np.save(OUT_DIR / "y.npy", np.array(y, dtype=np.int64))

# 3. On enregistre la correspondance entre les coups encodés et leur représentation réelle
with open(OUT_DIR / "move_to_int.pkl", "wb") as f:
    pickle.dump(move_to_int, f)

# endregion