import numpy as np

class RewardShaper:
    """Reward shaping avancé inspiré de Ng et al. 1999"""
    def __init__(self):
        self.last_hand_strength = 0
        self.history = []
        
    def shape(self, raw_reward, obs, done):
        """Transforme la récompense brute selon des heuristiques"""
        shaped_reward = raw_reward
        
        # 1. Pénalité pour dominos forts conservés (somme des points^2)
        hand_strength = np.sum([sum(d)**2 for d in obs['hand']])
        shaped_reward -= 0.01 * hand_strength
        
        # 2. Bonus pour diversité des options
        valid_actions = np.sum(obs['valid_actions'])
        shaped_reward += 0.1 * valid_actions
        
        # 3. Bonus pour création de doubles options
        left, right = obs['board'][0][0], obs['board'][-1][1]
        double_option_bonus = 1 if left == right else 0
        shaped_reward += 2 * double_option_bonus
        
        # 4. Pénalité pour répétition de motifs
        if len(self.history) > 3:
            if self.history[-3:] == self.history[-6:-3]:
                shaped_reward -= 5
                
        self.history.append(obs['board'].copy())
        return shaped_reward