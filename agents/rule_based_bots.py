import random
from typing import List, Tuple, Dict, Any, Optional
from environment.domino_env import DominoEnv
from environment.utils import ALL_DOMINOS, INDEX_TO_DOMINO, MAX_DOMINO_VALUE
from environment.domino_tile import DominoTile 
import numpy as np
import logging 

logger = logging.getLogger(__name__)

class RandomBot:
    """Un bot qui joue un coup légal aléatoire."""
    def select_action(self, observation: Dict[str, Any], legal_action_mask: np.ndarray) -> int:
        """Sélectionne une action aléatoire parmi les actions légales."""
        legal_indices = np.where(legal_action_mask)[0]
        if len(legal_indices) == 0:
            logger.error("RandomBot: Aucune action légale trouvée dans le masque ! Le jeu devrait être terminé ou il y a une erreur.")
            return 0 
        return random.choice(legal_indices)

class GreedyBot:
    """
    Un bot qui essaie de jouer le domino avec la plus grande somme de points.
    Priorité: Jouer Max Valeur > Piocher > Passer.
    """

    def _decode_hand(self, hand_encoding: np.ndarray) -> List[DominoTile]:
        """ Reconstruit la liste des tuiles de la main à partir de l'encodage multi-binaire. """
        hand = []
        if hand_encoding.shape[0] == len(ALL_DOMINOS):
            for i, present in enumerate(hand_encoding):
                if present > 0.5: # Utiliser > 0.5 pour comparer les floats/binaires
                    hand.append(INDEX_TO_DOMINO[i])
        else:
             logger.error(f"GreedyBot: Taille de hand_encoding ({hand_encoding.shape[0]}) incompatible avec ALL_DOMINOS ({len(ALL_DOMINOS)})")
        # hand.sort() 
        return hand


    def select_action(self, observation: Dict[str, Any], legal_action_mask: np.ndarray) -> int:
        """Sélectionne l'action 'la plus gourmande' parmi les actions légales."""

        action_dim = len(legal_action_mask)
        action_draw = action_dim - 2
        action_pass = action_dim - 1

        legal_indices = np.where(legal_action_mask)[0]

        if len(legal_indices) == 0:
            logger.error("GreedyBot: Aucune action légale trouvée dans le masque!")
            return action_pass if legal_action_mask[action_pass] else (action_draw if legal_action_mask[action_draw] else 0)


        best_play_action = -1
        max_tile_value = -1
        can_draw = False
        can_pass = False

        my_hand: List[DominoTile] = []
        if 'my_hand_encoding' in observation: 
             my_hand = self._decode_hand(observation['my_hand_encoding'])
        elif 'my_hand' in observation and isinstance(observation['my_hand'][0], DominoTile): # Si l'obs contient la liste
             my_hand = observation['my_hand']
        else:
             logger.warning("GreedyBot: Impossible de déterminer la main du joueur depuis l'observation.")
             return random.choice(legal_indices)


        for action_code in legal_indices:
            if action_code == action_draw:
                can_draw = True
                continue 
            if action_code == action_pass:
                can_pass = True
                continue 

            if best_play_action == -1: 
                 best_play_action = action_code

        if best_play_action != -1:
             logger.debug(f"GreedyBot (simplifié): Choisit l'action 'play' {best_play_action}")
             return best_play_action
        elif can_draw:
            logger.debug(f"GreedyBot: Choisit de piocher ({action_draw})")
            return action_draw
        elif can_pass:
            logger.debug(f"GreedyBot: Choisit de passer ({action_pass})")
            return action_pass
        else:
            logger.error("GreedyBot: Aucune action logique trouvée après évaluation (play/draw/pass).")
            return random.choice(legal_indices)