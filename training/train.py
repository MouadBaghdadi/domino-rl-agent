# training/train.py
import torch
import yaml
import os
import numpy as np
import logging
from tqdm import tqdm # Barre de progression

from environment.domino_env import DominoEnv
from agents.ppo_lstm_agent import PpoLstmAgent
from agents.rule_based_bots import RandomBot, GreedyBot
from training.curriculum import CurriculumManager
from training.reward_shaper import RewardShaper
from evaluation.evaluator import evaluate_agent
from torch.utils.tensorboard import SummaryWriter # Pour les logs Tensorboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'ppo_config.yaml')

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration chargée:")
    logger.info(config)

    log_dir = config.get('log_dir', './logs')
    model_save_dir = os.path.dirname(config.get('model_save_path', './models/ppo_domino_lstm'))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) 
    env = DominoEnv(render_mode=None) 

    obs_sample = env.observation_space.sample()
    obs_dim_approx = sum(np.prod(v.shape) if hasattr(v, 'shape') else 1 for k, v in obs_sample.items() if k != 'board_sequence') 
    obs_dim_approx = int(obs_dim_approx) 
    logger.info(f"Dimension approximative de l'observation (hors séquence): {obs_dim_approx}")
    action_dim = env.action_space.n
    agent = PpoLstmAgent(obs_dim_approx, action_dim, config)
    
    def create_target_network_copy(agent):
        """Create a copy of the agent for self-play with frozen weights."""
        target_agent = PpoLstmAgent(obs_dim_approx, action_dim, config)
        target_agent.load_model(agent.network.state_dict())  # Copy current weights
        target_agent.optimizer = None
        return target_agent
    
    target_network = create_target_network_copy(agent)
    curriculum = CurriculumManager(CONFIG_PATH)

    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(env),
        "self": target_network 
    }

    try:
        reward_shaper = RewardShaper(config.get('reward_shaping', {}))
    except Exception as e:
        logger.warning(f"Impossible d'initialiser le RewardShaper: {e}. Utilisation des récompenses standards.")
        class DummyRewardShaper:
            def shape_reward(self, reward, *args, **kwargs):
                return reward
        reward_shaper = DummyRewardShaper()

    total_timesteps = config.get('total_timesteps', 1000000)
    steps_per_collect = config.get('steps_per_collect', 2048)
    eval_frequency = config.get('eval_frequency', 20000)
    save_frequency = config.get('save_frequency', 50000)

    global_step = 0
    episode_num = 0
    last_eval_step = 0
    last_save_step = 0

    pbar = tqdm(total=total_timesteps, desc="Training Progress")

    def update_target_network(agent, target_agent, tau=1.0):
        """
        Update target network weights with current agent weights.
        tau: interpolation parameter (1.0 = complete replacement, <1.0 = partial update)
        """
        if tau == 1.0:
            target_agent.load_model(agent.network.state_dict())
        else:
            for target_param, param in zip(target_agent.network.parameters(), agent.network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    while global_step < total_timesteps:

        opponent_type = curriculum.get_opponent_type(global_step)
        opponent = bots[opponent_type]
        if opponent_type == "self" and isinstance(opponent, PpoLstmAgent):
            
            target_update_frequency = config.get('target_update_frequency', 10000)  
            if global_step % target_update_frequency == 0:
                logger.info(f"Updating target network for self-play at step {global_step}")
                tau = config.get('target_update_tau', 1.0) 
                update_target_network(agent, opponent, tau)
        
        agent.memory.clear() 
        steps_collected = 0
        episode_rewards = [] 
        episode_lengths = []

        while steps_collected < steps_per_collect:
            episode_reward = 0
            episode_length = 0
            obs_dict, info = env.reset()
            agent.reset() 
            if opponent_type != 'self' and hasattr(opponent, 'reset'):
                 opponent.reset()

            terminated = truncated = False
            current_player = info.get('current_player', 0) 
            while not (terminated or truncated):
                acting_agent = agent if current_player == 0 else opponent
                other_agent = opponent if current_player == 0 else agent

                legal_mask = env.get_legal_action_mask()

                if isinstance(acting_agent, PpoLstmAgent):
                    action, log_prob, value = acting_agent.select_action(obs_dict, legal_mask)
                elif isinstance(acting_agent, (RandomBot, GreedyBot)):
                    action = acting_agent.select_action(obs_dict, legal_mask)
                    log_prob, value = 0.0, 0.0 
                else:
                     logger.error(f"Type d'agent inconnu: {type(acting_agent)}")
                     action = env.action_space.sample() 
                     log_prob, value = 0.0, 0.0

                next_obs_dict, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if current_player == 0:
                    try:
                        shaped_reward = reward_shaper.shape_reward(reward, env._decode_action(action), info)
                    except (AttributeError, TypeError):
                        shaped_reward = reward
                    
                    agent.store_transition(obs_dict, action, shaped_reward, done, log_prob, value, next_obs_dict)
                    episode_reward += shaped_reward 
                    steps_collected += 1
                    global_step += 1
                    pbar.update(1)

                obs_dict = next_obs_dict
                current_player = info.get('current_player') 
                episode_length += 1

                if done or steps_collected >= steps_per_collect:
                    break

            episode_num += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            logger.debug(f"Episode {episode_num} fini. Récompense: {episode_reward}, Longueur: {episode_length}")


        if len(agent.memory) > 0:
            logger.info(f"Global Step: {global_step}. Lancement de l'apprentissage PPO avec {len(agent.memory)} transitions.")
            agent.learn() 
            logger.info("Apprentissage PPO terminé.")
        else:
             logger.warning("Aucune transition collectée pour l'apprentissage.")


        if global_step >= last_eval_step + eval_frequency:
            logger.info(f"Global Step: {global_step}. Lancement de l'évaluation...")
            avg_reward = evaluate_agent(agent, config, return_avg_reward_only=True) # Get just the average reward
            logger.info(f"Évaluation terminée. Récompense moyenne: {avg_reward:.2f}")
            writer.add_scalar('Evaluation/AverageReward', avg_reward, global_step)
            last_eval_step = global_step

        if global_step >= last_save_step + save_frequency:
            save_path = f"{config['model_save_path']}_step{global_step}.pth"
            logger.info(f"Global Step: {global_step}. Sauvegarde du modèle sur {save_path}")
            agent.save_model(save_path)
            last_save_step = global_step

        if episode_rewards:
             avg_ep_reward = np.mean(episode_rewards)
             avg_ep_length = np.mean(episode_lengths)
             logger.info(f"Stats Collecte - Avg Reward: {avg_ep_reward:.2f}, Avg Length: {avg_ep_length:.1f}")
             writer.add_scalar('Training/AverageEpisodeReward', avg_ep_reward, global_step)
             writer.add_scalar('Training/AverageEpisodeLength', avg_ep_length, global_step)


    pbar.close()
    env.close()
    writer.close()
    logger.info("Entraînement terminé.")
    final_save_path = f"{config['model_save_path']}_final.pth"
    agent.save_model(final_save_path)
    logger.info(f"Modèle final sauvegardé sur {final_save_path}")

if __name__ == "__main__":
    main()