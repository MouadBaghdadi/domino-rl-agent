import numpy as np
from environment.domino_env import DominoEnv

class CurriculumWrapper:
    """Implémente le curriculum learning basé sur Narvekar et al. 2020"""
    def __init__(self):
        self.env = DominoEnv()
        self.difficulty_levels = [
            {'visible_hands': True, 'blocking_enabled': False},  # Niveau 0
            {'visible_hands': False, 'blocking_enabled': False}, # Niveau 1
            {'visible_hands': False, 'blocking_enabled': True}   # Niveau 2
        ]
        self.current_level = 0
        self.win_threshold = 0.8  # Taux de victoire requis pour niveau suivant
        
    def _update_difficulty(self, win_rate):
        if win_rate > self.win_threshold and self.current_level < 2:
            self.current_level += 1
            print(f"Passage au niveau de difficulté {self.current_level}")
            
    def get_obs(self, raw_obs):
        """Modifie les observations selon le niveau de difficulté"""
        config = self.difficulty_levels[self.current_level]
        
        if config['visible_hands']:
            # Révèle la main de l'adversaire
            raw_obs['hidden_state'] = np.ones(28)
            opponent_hand = self.env.players[1 - self.env.current_player]
            for d in opponent_hand:
                raw_obs['hidden_state'][self.env.domino_to_id(d)] = 0.0
                
        return raw_obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Désactive le blocage pour les premiers niveaux
        if not self.difficulty_levels[self.current_level]['blocking_enabled']:
            if self.env._is_blocked():
                done = False
                reward += 10  # Encourage à continuer
                
        return self.get_obs(obs), reward, done, info
    
    def reset(self):
        return self.get_obs(self.env.reset())