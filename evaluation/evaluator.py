import numpy as np
from tqdm import tqdm
from environment.domino_env import DominoEnv
from agents.ppo_agent import PPOTrainer
from agents.rule_based_bots import RandomBot, GreedyBot, DefensiveBot
from multiprocessing import Pool

class Evaluator:
    def __init__(self, model_path="models/best_model.pt"):
        self.env = DominoEnv()
        self.agent = PPOTrainer(self.env)
        self.agent.load_checkpoint(model_path)
        self.bots = {
            "random": RandomBot(),
            "greedy": GreedyBot(),
            "defensive": DefensiveBot()  # À ajouter dans rule_based_bots.py
        }

    def _run_match(self, opponent_type, num_games=100):
        """Exécute une série de matchs contre un type d'adversaire"""
        wins = 0
        for _ in range(num_games):
            obs = self.env.reset()
            done = False
            current_player = 0
            
            while not done:
                if current_player == 0:  # Notre agent
                    action = self.agent.get_action(obs, training=False)
                else:  # Bot
                    action = self.bots[opponent_type].act(obs)
                
                obs, _, done, _ = self.env.step(action)
                current_player = 1 - current_player
            
            if self.env.winner == 0:
                wins += 1
                
        return wins / num_games

    def benchmark(self, num_games=500, parallel=True):
        """Évaluation multi-processus avec barre de progression"""
        results = {}
        
        if parallel:
            with Pool() as p:
                tasks = [(bot,) for bot in self.bots.keys()]
                win_rates = list(tqdm(
                    p.starmap(self._run_match, tasks),
                    total=len(tasks),
                    desc="Benchmarking"
                ))
                results = dict(zip(self.bots.keys(), win_rates))
        else:
            for bot_name in tqdm(self.bots, desc="Benchmarking"):
                results[bot_name] = self._run_match(bot_name, num_games)
        
        print("\n=== Résultats finaux ===")
        for bot, wr in results.items():
            print(f"{bot}: {wr:.2%}")
            
        return results