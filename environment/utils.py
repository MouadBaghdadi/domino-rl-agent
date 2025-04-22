import torch
from typing import List
from .domino_tile import DominoTile

MAX_DOMINO_VALUE = 6
NUM_DOMINOS = (MAX_DOMINO_VALUE + 1) * (MAX_DOMINO_VALUE + 2) // 2
HAND_SIZE = 7
TOTAL_TILES = 28 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_all_dominos(max_value: int = MAX_DOMINO_VALUE) -> List[DominoTile]:
    """Génère la liste complète des dominos."""
    tiles = []
    for i in range(max_value + 1):
        for j in range(i, max_value + 1):
            tiles.append(DominoTile(i, j))
    return tiles

ALL_DOMINOS = generate_all_dominos()
DOMINO_TO_INDEX = {tile: i for i, tile in enumerate(ALL_DOMINOS)}
INDEX_TO_DOMINO = {i: tile for i, tile in enumerate(ALL_DOMINOS)}
