import numpy as np
from typing import List, Set, Dict, Optional, Tuple 
from environment.domino_tile import DominoTile
from environment.utils import MAX_DOMINO_VALUE, ALL_DOMINOS, DOMINO_TO_INDEX, INDEX_TO_DOMINO

class BeliefNetwork:
    """
    Estime la probabilité que l'adversaire possède chaque domino.
    Prend en compte la main de l'agent, le plateau, et les actions de l'adversaire.
    """
    def __init__(self):
        # self.possible_opponent_hands: Optional[Set[frozenset[DominoTile]]] = None

        self.tile_probabilities = np.zeros(len(ALL_DOMINOS))
        # self.known_opponent_tiles: Set[DominoTile] = set()
        self.known_non_opponent_tiles: Set[DominoTile] = set()
        self.reset()

    def reset(self):
        """Réinitialise les croyances au début d'une partie."""
        self.tile_probabilities.fill(0.0)
        # self.known_opponent_tiles = set() # Supprimé
        self.known_non_opponent_tiles = set()


    def _get_tile_index(self, tile: DominoTile) -> Optional[int]:
        """Trouve l'index global d'une tuile."""
        return DOMINO_TO_INDEX.get(tile, None)


    def _update_known_sets(self, my_hand: List[DominoTile], board: List[DominoTile]):
        """Met à jour l'ensemble des tuiles qui ne sont PAS chez l'adversaire."""
        self.known_non_opponent_tiles.update(my_hand)
        self.known_non_opponent_tiles.update(board)

        for tile in self.known_non_opponent_tiles:
            idx = self._get_tile_index(tile)
            if idx is not None:
                self.tile_probabilities[idx] = 0.0

    def _normalize_probabilities(self, current_opponent_hand_size: int):
        """
        Normalise les probabilités des tuiles *inconnues* pour qu'elles
        somment à la taille actuelle connue de la main de l'adversaire.
        """
        if current_opponent_hand_size <= 0:
            self.tile_probabilities.fill(0.0) 
            return

        unknown_indices = []
        for i, tile in enumerate(ALL_DOMINOS):
            if tile not in self.known_non_opponent_tiles:
                unknown_indices.append(i)

        if not unknown_indices:
             if current_opponent_hand_size > 0:
                  print(f"WARN (BeliefNetwork): Incohérence - aucune tuile inconnue mais taille main adverse = {current_opponent_hand_size}")
             self.tile_probabilities.fill(0.0) 
             for tile in self.known_non_opponent_tiles:
                 idx = self._get_tile_index(tile)
                 if idx is not None: self.tile_probabilities[idx] = 0.0
             return


        current_prob_sum_unknown = self.tile_probabilities[unknown_indices].sum()

        if current_prob_sum_unknown <= 1e-9: 
             if current_opponent_hand_size > 0:
                  print(f"WARN (BeliefNetwork): Somme probas inconnues nulle, réinitialisation uniforme pour {current_opponent_hand_size} tuiles.")
                  uniform_prob = current_opponent_hand_size / len(unknown_indices)
                  temp_probs = np.zeros_like(self.tile_probabilities)
                  temp_probs[unknown_indices] = uniform_prob
                  self.tile_probabilities = temp_probs
             else: 
                  self.tile_probabilities.fill(0.0)

        else:
            scale_factor = current_opponent_hand_size / current_prob_sum_unknown
            unknown_mask = np.zeros_like(self.tile_probabilities, dtype=bool)
            unknown_mask[unknown_indices] = True

            self.tile_probabilities[unknown_mask] *= scale_factor

            known_mask = ~unknown_mask
            self.tile_probabilities[known_mask] = 0.0


        self.tile_probabilities = np.clip(self.tile_probabilities, 0.0, 1.0)

        # final_sum = self.tile_probabilities[unknown_indices].sum()
        # if not np.isclose(final_sum, current_opponent_hand_size):
        #     print(f"DEBUG: Normalisation finale - Somme: {final_sum}, Attendu: {current_opponent_hand_size}")


    def update(self,
               my_hand: List[DominoTile],
               board: List[DominoTile],
               opponent_action_info: Tuple[str, Optional[DominoTile], Optional[int]],
               open_ends: List[int],
               opponent_hand_size: int):
        """
        Met à jour les croyances après une action de l'adversaire ou au début.

        Args:
            my_hand: Liste des tuiles dans la main de l'agent.
            board: Liste des tuiles actuellement sur le plateau.
            opponent_action_info: Tuple décrivant l'action de l'adversaire:
                                 ('play', tile_played, end_matched)
                                 ('draw', None, None)
                                 ('pass', None, None)
                                 ('start', None, None) -> Action virtuelle pour la première mise à jour
            open_ends: Liste des valeurs numériques aux extrémités ouvertes du plateau.
            opponent_hand_size: Nombre actuel de tuiles dans la main de l'adversaire (connu via l'env).
        """
        action_type, tile_played, _ = opponent_action_info

        is_first_update = not self.known_non_opponent_tiles
        if is_first_update:
            self._update_known_sets(my_hand, board)
            for i, tile in enumerate(ALL_DOMINOS):
                if tile not in self.known_non_opponent_tiles:
                    self.tile_probabilities[i] = 1.0

        else: 
            self._update_known_sets(my_hand, board) 

            if action_type == 'play' and tile_played is not None:
                pass

            elif action_type == 'draw':
                pass

            elif action_type == 'pass':
                indices_to_zero = []
                for i, prob in enumerate(self.tile_probabilities):
                    if prob > 1e-9: # Utiliser une petite tolérance
                        tile = INDEX_TO_DOMINO[i]
                        if any(tile.matches(end) for end in open_ends):
                            indices_to_zero.append(i)

                if indices_to_zero:
                    self.tile_probabilities[indices_to_zero] = 0.0
                    # print(f"DEBUG (BeliefNetwork): Pass - Mise à zéro probas pour indices {indices_to_zero}")


        self._normalize_probabilities(opponent_hand_size)


    def get_probabilities(self) -> np.ndarray:
        """Retourne le vecteur de probabilités (ou d'espérance) pour chaque tuile."""
        return self.tile_probabilities.copy()


    def get_most_likely_tiles(self, n: int = 5) -> List[Tuple[DominoTile, float]]:
        """Retourne les n tuiles les plus probables et leur probabilité estimée."""
        sorted_indices = np.argsort(-self.tile_probabilities)

        top_n = []
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = self.tile_probabilities[idx]
            if prob > 1e-9:
                top_n.append((INDEX_TO_DOMINO[idx], prob))
            else:
                break
        return top_n
