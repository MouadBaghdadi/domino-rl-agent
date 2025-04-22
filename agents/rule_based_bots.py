import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from environment.domino_env import DominoEnv 
class RandomBot:
    """Un bot qui joue un coup légal aléatoire."""
    def select_action(self, observation: Dict[str, Any], legal_action_mask: np.ndarray) -> int:
        legal_indices = np.where(legal_action_mask)[0]
        if len(legal_indices) == 0:
             print("WARN: RandomBot n'a trouvé aucune action légale via le masque!")
             return 0 
        return random.choice(legal_indices)

class GreedyBot:
    """Un bot qui essaie de jouer le domino avec la plus grande valeur."""
    def __init__(self, env: DominoEnv):
         self.env = env

    def select_action(self, observation: Dict[str, Any], legal_action_mask: np.ndarray) -> int:
        legal_actions_map = self.env._get_legal_actions_encoded() 
        best_action_code = -1
        max_value = -1
        can_draw = False
        pass_action_code = -1

        my_hand = observation['my_hand'] 
        for action_code, action_details in legal_actions_map.items():
            if action_details["type"] == "play":
                 tile_idx = action_details["tile_idx"]
                 try:
                     current_hand = self.env.player_hands[self.env.current_player]
                     tile = current_hand[tile_idx]
                     tile_value = tile.get_value()
                     if tile_value > max_value:
                         max_value = tile_value
                         best_action_code = action_code
                 except (IndexError, AttributeError):
                      print(f"WARN: GreedyBot ne peut pas récupérer la tuile {tile_idx} de la main.")
                      continue

            elif action_details["type"] == "draw":
                can_draw = True
                draw_action_code = action_code
            elif action_details["type"] == "pass":
                pass_action_code = action_code

        if best_action_code != -1:
            return best_action_code 
        elif can_draw:
              legal_indices = np.where(legal_action_mask)[0]
              non_play_actions = [idx for idx in legal_indices if legal_actions_map.get(idx, {}).get("type") != "play"]
              if non_play_actions: return non_play_actions[0]

        elif pass_action_code != -1:
             return pass_action_code 
        else:
              print("WARN: GreedyBot n'a trouvé aucune action logique!")
              legal_indices = np.where(legal_action_mask)[0]
              return random.choice(legal_indices) if len(legal_indices)>0 else 0


