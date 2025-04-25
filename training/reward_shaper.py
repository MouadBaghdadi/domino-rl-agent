from typing import Dict, Any, Optional
from environment.domino_tile import DominoTile

class RewardShaper:
    """
    Classe pour modifier la récompense de base fournie par l'environnement
    en ajoutant des termes de récompense/pénalité intermédiaires.

    ATTENTION: Le reward shaping peut être délicat. Des récompenses intermédiaires
    mal conçues peuvent conduire l'agent à optimiser ces récompenses artificielles
    au détriment de l'objectif réel (gagner la partie). À utiliser avec prudence.
    """

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        """
        Initialise le RewardShaper avec une configuration initiale.

        Args:
            initial_config: Un dictionnaire contenant les paramètres de shaping.
                            Exemple: {'enabled': True, 'penalty_draw': -0.01, ...}
                            Si None, le shaping est désactivé.
        """
        self.config = initial_config if initial_config else {'enabled': False}
        self._update_params() 

    def _update_params(self):
        """Met à jour les paramètres de shaping internes à partir de self.config."""
        self.enabled = self.config.get('enabled', False)
        self.penalty_draw = self.config.get('penalty_draw', 0.0) 
        self.reward_play_double = self.config.get('reward_play_double', 0.1)
        self.penalty_illegal_pass = self.config.get('penalty_illegal_pass', -0.1)
        self.reward_empty_hand_fast = self.config.get('reward_empty_hand_fast', 0.1)
        self.penalty_double_not_played = self.config.get('penalty_double_not_played', -0.1)
        self.illegal_pass_types = {"pass_illegal_can_play", "pass_illegal_must_draw"}

    def update_config(self, new_config: Optional[Dict[str, Any]]):
        """
        Met à jour la configuration du reward shaping. Appelée par la boucle
        d'entraînement si le curriculum change les paramètres de shaping.

        Args:
            new_config: Le nouveau dictionnaire de configuration, ou None pour désactiver.
        """
        self.config = new_config if new_config else {'enabled': False}
        self._update_params() 

    def shape_reward(self,
                     base_reward: float,
                     info_dict: Optional[Dict[str, Any]] = None 
                    ) -> float:
        """
        Modifie la récompense de base en fonction de l'action effectuée et de la config.

        Args:
            base_reward: La récompense fournie par l'environnement (ex: +1, -1, 0).
            info_dict: Le dictionnaire d'informations retourné par env.step(). Doit
                       contenir une clé 'action_details' avec les infos sur l'action
                       tentée/effectuée (ex: {'type': 'play', 'tile': ...}).

        Returns:
            La récompense modifiée (shapée).
        """
        if not self.enabled or info_dict is None:
            return base_reward

        shaped_reward = base_reward
        action_details = info_dict.get("action_details")

        if not isinstance(action_details, dict):
             print("[RewardShaper] Warning: Missing or invalid 'action_details' in info_dict.") 
             return base_reward

        action_type = action_details.get("type")

        if action_type == "draw": 
            shaped_reward += self.penalty_draw

        elif action_type == "play":
            tile_played = action_details.get("tile")
            if isinstance(tile_played, DominoTile) and tile_played.is_double():
                 shaped_reward += self.reward_play_double

        elif action_type in self.illegal_pass_types:
             shaped_reward += self.penalty_illegal_pass

        return shaped_reward