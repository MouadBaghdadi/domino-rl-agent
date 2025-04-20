import numpy as np
from environment.domino_env import DominoEnv

class RandomBot:
    """Joue des coups aléatoires valides"""
    def act(self, obs):
        valid_actions = np.where(obs['valid_actions'] == 1)[0]
        return np.random.choice(valid_actions)

class GreedyBot:
    """Joue toujours le domino avec le plus de points"""
    def act(self, obs):
        valid_ids = np.where(obs['valid_actions'] == 1)[0]
        dominoes = [DominoEnv.id_to_domino(id) for id in valid_ids]
        scores = [sum(d) for d in dominoes]
        return valid_ids[np.argmax(scores)]

class DefensiveBot:
    """Privilégie les dominos avec valeurs uniques"""
    def act(self, obs):
        valid_ids = np.where(obs['valid_actions'] == 1)[0]
        dominoes = [DominoEnv.id_to_domino(id) for id in valid_ids]
        
        # Score = somme des valeurs - 2 * max(valeurs uniques)
        hand = obs['hand']
        unique_counts = {i: np.sum(hand[:, i]) for i in range(7)}
        scores = []
        for d in dominoes:
            score = sum(d) - 2 * (unique_counts[d[0]] + unique_counts[d[1]])
            scores.append(score)
            
        return valid_ids[np.argmax(scores)]