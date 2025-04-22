import numpy as np
from typing import List, Set, Dict, Optional, Tuple
from environment.domino_tile import DominoTile
from environment.utils import MAX_DOMINO_VALUE, ALL_DOMINOS, DOMINO_TO_INDEX

class BeliefNetwork:
    """
    Estime la probabilité que l'adversaire possède chaque domino.
    Prend en compte la main de l'agent, le plateau, et les actions de l'adversaire.
    """
    def __init__(self, num_opponent_tiles: int):
        self.num_opponent_tiles_initial = num_opponent_tiles
        self.possible_opponent_hands: Optional[Set[frozenset[DominoTile]]] = None 

        self.tile_probabilities = np.zeros(len(ALL_DOMINOS))
        self.known_opponent_tiles: Set[DominoTile] = set()
        self.known_non_opponent_tiles: Set[DominoTile] = set() # Ma main + plateau

        self.reset()

    def reset(self):
        """Réinitialise les croyances au début d'une partie."""
        self.tile_probabilities = np.ones(len(ALL_DOMINOS)) 
        self.known_opponent_tiles = set()
        self.known_non_opponent_tiles = set()
        self._normalize_probabilities()


    def _get_tile_index(self, tile: DominoTile) -> Optional[int]:
        """Trouve l'index global d'une tuile."""
        DOMINO_TO_INDEX[tile]

    def _update_known_sets(self, my_hand: List[DominoTile], board: List[DominoTile]):
        """Met à jour l'ensemble des tuiles qui ne sont PAS chez l'adversaire."""
        self.known_non_opponent_tiles.update(my_hand)
        self.known_non_opponent_tiles.update(board)

        # Mettre à zéro les probabilités des tuiles connues hors de la main adverse
        for tile in self.known_non_opponent_tiles:
            idx = self._get_tile_index(tile)
            if idx is not None:
                self.tile_probabilities[idx] = 0.0

    def _normalize_probabilities(self):
        """Normalise les probabilités pour qu'elles somment au nombre supposé de tuiles adverses."""
        current_opponent_hand_size = self.get_estimated_opponent_hand_size()
        if current_opponent_hand_size <= 0:
            self.tile_probabilities.fill(0.0)
            return

        valid_probs = self.tile_probabilities[self.tile_probabilities > 0]
        if valid_probs.sum() > 0:
            self.tile_probabilities[self.tile_probabilities > 0] = (valid_probs / valid_probs.sum()) * current_opponent_hand_size
            self.tile_probabilities = np.clip(self.tile_probabilities, 0.0, 1.0)


    def update_belief(self, my_hand: List[DominoTile], board: List[DominoTile], opponent_action: Tuple[str, Optional[DominoTile], Optional[int]], open_ends: List[int], opponent_hand_size: int):
        """
        Met à jour les croyances après une action de l'adversaire.
        opponent_action: ('play', tile, end) ou ('draw', None, None) ou ('pass', None, None)
        """
        self._update_known_sets(my_hand, board)

        action_type, tile_played, _ = opponent_action

        if action_type == 'play' and tile_played is not None:
            idx = self._get_tile_index(tile_played)
            if idx is not None:
                self.tile_probabilities[idx] = 0.0
            self.known_non_opponent_tiles.add(tile_played)

        elif action_type == 'draw':
            # Piocher ne révèle rien directement sur les tuiles spécifiques,
            # mais augmente la taille de la main et donc la probabilité moyenne.
            # La normalisation s'en chargera.
            pass

        elif action_type == 'pass':
            possible_plays = []
            for i, prob in enumerate(self.tile_probabilities):
                if prob > 0: 
                    tile = ALL_DOMINOS[i]
                    if any(tile.matches(end) for end in open_ends):
                        possible_plays.append(i)

            if possible_plays:
                self.tile_probabilities[possible_plays] *= 0.1 # Réduire la probabilité des tuiles qui ne peuvent pas être jouées

        self._normalize_probabilities()


    def get_probabilities(self) -> np.ndarray:
        """Retourne le vecteur de probabilités pour chaque tuile."""
        return self.tile_probabilities.copy()

    def get_estimated_opponent_hand_size(self) -> int:
        """Estime la taille de la main adverse basée sur les probabilités non nulles."""
        # return int(np.sum(self.tile_probabilities > 1e-6)) # Compter les > 0
        return int(round(np.sum(self.tile_probabilities))) 

    def get_most_likely_tiles(self, n=5) -> List[Tuple[DominoTile, float]]:
        """Retourne les n tuiles les plus probables et leur probabilité estimée."""
        indexed_probs = list(enumerate(self.tile_probabilities))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        top_n = []
        for i in range(min(n, len(indexed_probs))):
            idx, prob = indexed_probs[i]
            if prob > 0: 
                top_n.append((ALL_DOMINOS[idx], prob))
        return top_n