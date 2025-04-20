import torch
import numpy as np
from tqdm import tqdm
from agents.ppo_agent import PPOTrainer
from training.curriculum import CurriculumWrapper
from training.reward_shaper import RewardShaper

class Trainer:
    def __init__(self):
        self.curriculum = CurriculumWrapper()
        self.shaper = RewardShaper()
        self.agent = PPOTrainer(self.curriculum.env)
        
        # Configuration
        self.total_episodes = 10000
        self.eval_interval = 500
        self.batch_size = 4096
        
    def _run_episode(self):
        """Exécute un épisode complet avec curriculum et reward shaping"""
        obs = self.curriculum.reset()
        episode_data = []
        
        while True:
            action = self.agent.get_action(obs)
            next_obs, reward, done, _ = self.curriculum.step(action)
            
            # Application du reward shaping
            shaped_reward = self.shaper.shape(reward, obs, done)
            
            # Stockage des données
            episode_data.append((obs, action, shaped_reward, done))
            
            if done:
                # Mise à jour du curriculum
                self.curriculum._update_difficulty(self.agent.win_rate)
                return episode_data
            
            obs = next_obs
            
    def train(self):
        """Boucle d'entraînement principale"""
        progress = tqdm(total=self.total_episodes)
        
        for episode in range(self.total_episodes):
            # Collecte des expériences
            batch = []
            for _ in range(self.batch_size):
                batch.extend(self._run_episode())
            
            # Entraînement du modèle
            self.agent.update_policy(batch)
            
            # Évaluation périodique
            if episode % self.eval_interval == 0:
                win_rates = self.agent.evaluate()
                progress.set_description(
                    f"Level {self.curriculum.current_level} | "
                    f"WR vs Random: {win_rates['RandomBot']:.2%} | "
                    f"WR vs Greedy: {win_rates['GreedyBot']:.2%}"
                    f" | WR vs Defensive: {win_rates['DefensiveBot']:.2%}"
                )
                
            progress.update(1)
            
        progress.close()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()