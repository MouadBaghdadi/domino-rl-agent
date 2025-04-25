import numpy as np
from typing import List, Dict, Optional

import torch 
from .domino_tile import DominoTile


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_DOMINO_VALUE: int = 6
"""Valeur maximale sur un côté d'un domino."""

NUM_DOMINOS: int = (MAX_DOMINO_VALUE + 1) * (MAX_DOMINO_VALUE + 2) // 2
"""Nombre total de dominos uniques dans un jeu de double-six (28)."""

HAND_SIZE: int = 7
"""Nombre initial de dominos distribués à chaque joueur."""

TOTAL_TILES: int = NUM_DOMINOS
"""Nombre total de tuiles dans le jeu (identique à NUM_DOMINOS pour double-six)."""


def generate_all_dominos(max_value: int = MAX_DOMINO_VALUE) -> List[DominoTile]:
    """
    Génère la liste complète et canoniquement triée des dominos uniques
    pour la valeur maximale donnée.

    Args:
        max_value: La valeur numérique maximale sur un côté de domino.

    Returns:
        Liste d'objets DominoTile uniques, triée.
    """
    tiles = []
    for i in range(max_value + 1):
        for j in range(i, max_value + 1): 
            tiles.append(DominoTile(i, j))
    return tiles

ALL_DOMINOS: List[DominoTile] = generate_all_dominos()
"""Liste précalculée de tous les dominos possibles (28), triée."""

DOMINO_TO_INDEX: Dict[DominoTile, int] = {tile: i for i, tile in enumerate(ALL_DOMINOS)}
"""Mapping précalculé d'un objet DominoTile vers son index entier (0-27)."""

INDEX_TO_DOMINO: Dict[int, DominoTile] = {i: tile for i, tile in enumerate(ALL_DOMINOS)}
"""Mapping précalculé d'un index entier (0-27) vers l'objet DominoTile correspondant."""


def decode_hand_from_encoding(hand_encoding: np.ndarray) -> List[DominoTile]:
    """
    Reconstruit la liste des tuiles de la main à partir d'un encodage
    multi-binaire (où l'index correspond à ALL_DOMINOS).

    Args:
        hand_encoding: Un array numpy de 0 ou 1, de taille NUM_DOMINOS.

    Returns:
        Une liste d'objets DominoTile présents dans la main.
        Retourne une liste vide si l'encodage est invalide.
    """
    hand = []
    if hand_encoding.shape == (NUM_DOMINOS,):
        for i, is_present in enumerate(hand_encoding):
            if is_present > 0.5:
                tile = INDEX_TO_DOMINO.get(i)
                if tile: 
                    hand.append(tile)
                else:
                     print(f"Erreur interne (decode_hand): Index {i} non trouvé dans INDEX_TO_DOMINO.")

    elif hand_encoding.size > 0: 
        print(f"Erreur (decode_hand): Taille d'encodage ({hand_encoding.shape}) incompatible avec NUM_DOMINOS ({NUM_DOMINOS}).")

    # hand.sort()
    return hand

