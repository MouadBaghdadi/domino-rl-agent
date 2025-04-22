from typing import Dict, Any

class RewardShaper:
    """Classe pour modifier ou ajouter des récompenses intermédiaires."""

    def __init__(self, config: Dict[str, Any]):
        self.penalty_per_draw = config.get('reward_shape_penalty_draw', -0.01)
        self.reward_play_double = config.get('reward_shape_play_double', 0.02)
        self.enabled = config.get('reward_shaping_enabled', False) # Activer/désactiver

    def shape_reward(self, base_reward: float, action_info: Dict[str, Any], state_info: Dict[str, Any]) -> float:
        """Modifie la récompense de base."""
        if not self.enabled:
            return base_reward

        shaped_reward = base_reward
        action_type = action_info.get("type")

        if action_type == "draw" or action_type == "forced_draw":
            shaped_reward += self.penalty_per_draw

        elif action_type == "play":
            tile_played = action_info.get("tile") 
            if tile_played and tile_played.is_double():
                shaped_reward += self.reward_play_double

        return shaped_reward