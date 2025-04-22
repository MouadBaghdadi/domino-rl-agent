import os
import torch
import yaml
import numpy as np
import logging
from typing import Dict, Any

from environment.domino_env import DominoEnv
from agents.ppo_lstm_agent import PpoLstmAgent
from agents.rule_based_bots import RandomBot, GreedyBot

logger = logging.getLogger(__name__)

def evaluate_agent(agent_path_or_agent, config_path_or_config, num_episodes: int = 50, opponent_type: str = "random", return_avg_reward_only: bool = False):
    """
    Évalue un agent entraîné contre un adversaire spécifique.

    Args:
        agent_path_or_agent: Chemin vers le fichier du modèle de l'agent PPO sauvegardé (.pth) ou une instance de PpoLstmAgent.
        config_path_or_config: Chemin vers le fichier de configuration YAML (pour les dims réseau) ou un dictionnaire de config.
        num_episodes: Nombre de parties à jouer pour l'évaluation.
        opponent_type: Type d'adversaire ('random', 'greedy', ou autre bot défini).
        return_avg_reward_only: Si True, retourne seulement la récompense moyenne (float) au lieu du dictionnaire complet.

    Returns:
        Dictionnaire contenant les métriques d'évaluation (ex: taux de victoire) ou float (avg_reward) si return_avg_reward_only=True.
    """
    if isinstance(config_path_or_config, dict):
        config = config_path_or_config
    else:
        with open(config_path_or_config, 'r') as f:
            config = yaml.safe_load(f)

    env = DominoEnv(render_mode=None) 

    obs_sample = env.observation_space.sample()
    obs_dim_approx = sum(np.prod(v.shape) if hasattr(v, 'shape') else 1 for k, v in obs_sample.items() if k != 'board_sequence')
    obs_dim_approx = int(obs_dim_approx)  # Convert to integer for LSTM
    action_dim = env.action_space.n

    if isinstance(agent_path_or_agent, PpoLstmAgent):
        agent = agent_path_or_agent
        logger.info("Utilisation de l'agent fourni directement")
    else:
        agent = PpoLstmAgent(obs_dim_approx, action_dim, config)
        try:
            agent.load_model(agent_path_or_agent)
            agent.network.eval() 
            logger.info(f"Agent chargé depuis {agent_path_or_agent}")
        except FileNotFoundError:
            logger.error(f"Fichier modèle non trouvé: {agent_path_or_agent}")
            return {"error": "Model file not found"}
        except Exception as e:
             logger.error(f"Erreur lors du chargement du modèle: {e}")
             return {"error": f"Failed to load model: {e}"}

    agent.network.eval()

    if opponent_type == "random":
        opponent = RandomBot()
    elif opponent_type == "greedy":
        opponent = GreedyBot(env)
    # elif opponent_type == "self": 
    #    opponent = PpoLstmAgent(...)
    else:
        logger.error(f"Type d'adversaire inconnu pour l'évaluation: {opponent_type}")
        return {"error": f"Unknown opponent type: {opponent_type}"}

    wins = 0
    losses = 0
    draws = 0
    total_reward_agent = 0

    logger.info(f"Début de l'évaluation contre {opponent_type} pour {num_episodes} épisodes...")

    for episode in range(num_episodes):
        obs_dict, info = env.reset()
        agent.reset()
        if hasattr(opponent, 'reset'): opponent.reset()
        terminated = truncated = False
        episode_reward_agent = 0 

        while not (terminated or truncated):
            current_player = env.current_player 

            acting_agent = agent if current_player == 0 else opponent
            legal_mask = env.get_legal_action_mask()

            with torch.no_grad():
                if isinstance(acting_agent, PpoLstmAgent):
                    action, _, _ = acting_agent.select_action(obs_dict, legal_mask)
                elif isinstance(acting_agent, (RandomBot, GreedyBot)):
                    action = acting_agent.select_action(obs_dict, legal_mask)
                else:
                    action = env.action_space.sample()

            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if current_player == 0:
                episode_reward_agent += reward

            obs_dict = next_obs_dict

            if done:
                winner = env.winner
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
                break 

        total_reward_agent += episode_reward_agent
        if (episode + 1) % 10 == 0:
            logger.info(f"Évaluation: Episode {episode + 1}/{num_episodes} terminé.")


    win_rate = wins / num_episodes if num_episodes > 0 else 0
    loss_rate = losses / num_episodes if num_episodes > 0 else 0
    draw_rate = draws / num_episodes if num_episodes > 0 else 0
    avg_reward = total_reward_agent / num_episodes if num_episodes > 0 else 0

    results = {
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "average_reward": avg_reward,
        "num_episodes": num_episodes,
        "opponent_type": opponent_type
    }
    logger.info(f"Résultats Évaluation: {results}")
    if return_avg_reward_only:
        return avg_reward
    else:
        return results

def main():
    MODEL_TO_EVAL = './models/ppo_domino_lstm_step102400.pth' 
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'ppo_config.yaml')
    N_EPISODES = 100
    OPPONENT = 'greedy' 

    if not os.path.exists(MODEL_TO_EVAL):
         print(f"Erreur: Fichier modèle non trouvé {MODEL_TO_EVAL}")
    elif not os.path.exists(CONFIG_FILE):
         print(f"Erreur: Fichier config non trouvé {CONFIG_FILE}")
    else:
        evaluation_results = evaluate_agent(MODEL_TO_EVAL, CONFIG_FILE, N_EPISODES, OPPONENT)
        print("\n--- Résultats de l'évaluation ---")
        for key, value in evaluation_results.items():
            print(f"{key}: {value}")
        print("-------------------------------\n")

if __name__ == "__main__":
    main()