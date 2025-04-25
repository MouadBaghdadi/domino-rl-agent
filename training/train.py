from typing import Any, Dict
import torch
import yaml
import os
import numpy as np
import logging
from tqdm import tqdm 
import copy 

from environment.domino_env import DominoEnv, TOTAL_ACTIONS
from agents.ppo_lstm_agent import PpoLstmAgent
from agents.rule_based_bots import RandomBot, GreedyBot
from environment.domino_tile import DominoTile
from training.curriculum import CurriculumManager
from training.reward_shaper import RewardShaper 
from evaluation.evaluator import evaluate_agent 
from torch.utils.tensorboard import SummaryWriter 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = 'configs/ppo_config.yaml'

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
    logger.info(f"Dimension approximative de l'observation (hors séquence): {obs_dim_approx}")
    action_dim = env.action_space.n
    logger.info(f"Dimension de l'espace d'action: {action_dim}")
    if action_dim != TOTAL_ACTIONS:
         logger.warning(f"ATTENTION: env.action_space.n ({action_dim}) ne correspond pas à la constante TOTAL_ACTIONS ({TOTAL_ACTIONS}) définie dans domino_env.py!")
    agent = PpoLstmAgent(action_dim=action_dim, config=config) 

    def create_target_network_copy(source_agent: PpoLstmAgent, config: Dict[str, Any], act_dim: int) -> PpoLstmAgent:
        """Create a copy of the agent for self-play with potentially frozen weights."""
        target_agent = PpoLstmAgent(action_dim=act_dim, config=config)
        assert source_agent.observation_dim == target_agent.observation_dim, \
               f"Dimension mismatch: Source({source_agent.observation_dim}) != Target({target_agent.observation_dim})"
        target_agent.network.load_state_dict(source_agent.network.state_dict())
        target_agent.network.eval() 
        # for param in target_agent.network.parameters():
        #     param.requires_grad = False
        logger.info("Target network created for self-play.")
        return target_agent

    def update_target_network(source_agent: PpoLstmAgent, target_agent: PpoLstmAgent, tau: float = 1.0):
        """
        Update target network weights with current agent weights.
        tau: interpolation parameter (1.0 = complete replacement, <1.0 = soft update)
        """
        if tau == 1.0:
            target_agent.network.load_state_dict(source_agent.network.state_dict()) 
            logger.debug("Target network updated (hard update).")
        else:
            source_state = source_agent.network.state_dict()
            target_state = target_agent.network.state_dict()
            for key in source_state:
                target_state[key] = tau * source_state[key] + (1.0 - tau) * target_state[key]
            target_agent.network.load_state_dict(target_state)
            logger.debug(f"Target network updated (soft update with tau={tau}).")

    curriculum = CurriculumManager(CONFIG_PATH)

    target_network = create_target_network_copy(agent, config, action_dim)

    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(), 
        "self": target_network 
    }

    try:
        reward_shaper_config = config.get('reward_shaping', {'enabled': False})
        reward_shaper = RewardShaper(reward_shaper_config)
        if reward_shaper.enabled:
            logger.info("Reward Shaper initialisé et activé.")
            reward_empty_hand_fast_factor = reward_shaper.reward_empty_hand_fast
            penalty_double_not_played_per_tile = reward_shaper.penalty_double_not_played
            FAST_WIN_TURN_THRESHOLD = config.get('fast_win_turn_threshold', 20)
        else:
            logger.info("Reward Shaper initialisé mais désactivé.")
            reward_empty_hand_fast_factor = 0.0
            penalty_double_not_played_per_tile = 0.0
            FAST_WIN_TURN_THRESHOLD = 999
    except Exception as e:
        logger.warning(f"Impossible d'initialiser le RewardShaper depuis la config: {e}. Shaping désactivé.")
        class DummyRewardShaper:
            def __init__(self, config=None): self.enabled = False
            def shape_reward(self, base_reward, *args, **kwargs): return base_reward
            def update_config(self, new_config): pass # Ne fait rien
        reward_shaper = DummyRewardShaper()
        reward_empty_hand_fast_factor = 0.0
        penalty_double_not_played_per_tile = 0.0
        FAST_WIN_TURN_THRESHOLD = 999

    total_timesteps = config.get('total_timesteps', 1000000)
    steps_per_collect = config.get('steps_per_collect', 2048)
    eval_frequency = config.get('eval_frequency', 20000)
    save_frequency = config.get('save_frequency', 50000)

    global_step = 0
    episode_num = 0
    last_eval_step = 0
    last_save_step = 0
    last_target_update_step = 0 
    pbar = tqdm(total=total_timesteps, desc="Training Progress")

    while global_step < total_timesteps:
        current_stage_config = curriculum.get_current_stage(global_step)
        opponent_type = current_stage_config.get('opponent', 'self') 
        opponent = bots[opponent_type]

        current_lr = curriculum.get_current_learning_rate(global_step, config.get('learning_rate', 3e-4))
        current_entropy = curriculum.get_current_entropy_coeff(global_step, config.get('entropy_coeff', 0.01))
        agent.update_hyperparameters(learning_rate=current_lr, entropy_coeff=current_entropy)
        writer.add_scalar('Curriculum/LearningRate', current_lr, global_step)
        writer.add_scalar('Curriculum/EntropyCoeff', current_entropy, global_step)

        current_shaping_config = curriculum.get_current_reward_shaping_config(global_step)
        if current_shaping_config is not None:
            reward_shaper.update_config(current_shaping_config)
            logger.info(f"Reward shaping config updated by curriculum: {current_shaping_config}")
        else:
            reward_shaper.update_config(config.get('reward_shaping', {'enabled': False})) 
            pass

        if opponent_type == "self":
            target_update_frequency = config.get('target_update_frequency', 10000)
            target_update_tau = config.get('target_update_tau', 1.0)
            if global_step >= last_target_update_step + target_update_frequency:
                logger.info(f"Updating target network for self-play at step {global_step} (tau={target_update_tau})")
                update_target_network(agent, opponent, target_update_tau) 
                last_target_update_step = global_step
            opponent.network.eval()

        agent.clear_memory()
        steps_collected = 0
        batch_episode_rewards = []
        batch_episode_lengths = []

        while steps_collected < steps_per_collect:
            episode_reward = 0
            episode_length = 0
            agent_player_index = 0
            episode_step_count_agent = 0

            obs_dict, info = env.reset()
            agent.reset()
            if opponent_type != 'self' and hasattr(opponent, 'reset'):
                 opponent.reset()

            terminated = truncated = False
            current_player = info.get('current_player_id', 0)

            while not (terminated or truncated):
                acting_agent = agent if current_player == agent_player_index else opponent
                other_agent = opponent if current_player == agent_player_index else agent

                legal_mask = env.get_legal_action_mask()

                if isinstance(acting_agent, PpoLstmAgent):
                    action, log_prob, value = acting_agent.select_action(obs_dict, info, legal_mask)
                elif isinstance(acting_agent, (RandomBot, GreedyBot)):
                    action = acting_agent.select_action(obs_dict, legal_mask)
                    log_prob, value = 0.0, 0.0
                else:
                     logger.error(f"Type d'agent inconnu: {type(acting_agent)}")
                     action = env.action_space.sample()
                     log_prob, value = 0.0, 0.0

                next_obs_dict, base_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                shaped_reward = base_reward

                if reward_shaper.enabled:
                    try:
                        step_shaped_reward = reward_shaper.shape_reward(base_reward, info)
                        shaped_reward = step_shaped_reward
                    except Exception as e:
                         logger.error(f"Erreur lors du reward shaping (étape): {e}. Utilisation récompense de base.")
                         shaped_reward = base_reward

                final_reward_adjustment = 0.0
                if done:
                    winner = info.get('winner', -1)
                    turn_count = info.get('turn_count', float('inf'))

                    if winner == agent_player_index:
                        if turn_count < FAST_WIN_TURN_THRESHOLD:
                             fast_win_bonus = reward_empty_hand_fast_factor * (FAST_WIN_TURN_THRESHOLD - turn_count)
                             final_reward_adjustment += fast_win_bonus
                    else:
                        agent_final_hand = info.get(f'player_{agent_player_index}_hand', [])
                        num_doubles_left = sum(1 for tile in agent_final_hand if isinstance(tile, DominoTile) and tile.is_double())
                        if num_doubles_left > 0:
                            double_penalty = num_doubles_left * penalty_double_not_played_per_tile
                            final_reward_adjustment += double_penalty

                    shaped_reward += final_reward_adjustment

                if current_player == agent_player_index:
                    preprocessed_obs_tensor = agent._preprocess_obs(obs_dict, info)
                    agent.store_transition(preprocessed_obs_tensor, action, shaped_reward, done, log_prob, value)

                    logger.debug(f"Step {global_step} | Ep {episode_num}: Agent action {action}, Shaped Reward {shaped_reward:.3f}")

                    episode_reward += shaped_reward
                    episode_step_count_agent += 1

                    steps_collected += 1
                    global_step += 1
                    pbar.update(1)

                obs_dict = next_obs_dict
                current_player = info.get('current_player_id', agent_player_index)
                episode_length += 1

                if done or steps_collected >= steps_per_collect:
                    break

            episode_num += 1
            if episode_step_count_agent > 0:
                 logger.info(f"Episode {episode_num} finished. Agent Reward: {episode_reward:.3f}, Total Length: {episode_length}, Agent Steps: {episode_step_count_agent}")

            batch_episode_rewards.append(episode_reward)
            batch_episode_lengths.append(episode_length)

        if len(agent.memory) > 0:
            logger.info(f"Global Step: {global_step}. Lancement de l'apprentissage PPO avec {len(agent.memory)} transitions.")
            agent.learn() 
            logger.info("Apprentissage PPO terminé.")
            if hasattr(agent, 'last_train_metrics') and agent.last_train_metrics:
                for key, val in agent.last_train_metrics.items():
                     writer.add_scalar(f'PPO/{key}', val, global_step)
        else:
             logger.warning("Aucune transition collectée pour l'apprentissage PPO (joueur 0 n'a peut-être pas joué).")

        if global_step >= last_eval_step + eval_frequency:
            logger.info(f"Global Step: {global_step}. Lancement de l'évaluation...")
            avg_reward = evaluate_agent(agent, config, num_episodes=config.get('eval_episodes', 20), return_avg_reward_only=True)
            if avg_reward is not None:
                 logger.info(f"Évaluation terminée. Récompense moyenne: {avg_reward:.2f}")
                 writer.add_scalar('Evaluation/AverageReward', avg_reward, global_step)
            else:
                 logger.warning("Évaluation n'a pas retourné de récompense moyenne.")
            last_eval_step = global_step

        if global_step >= last_save_step + save_frequency:
            save_path = f"{config['model_save_path']}_step{global_step}.pth"
            logger.info(f"Global Step: {global_step}. Sauvegarde du modèle sur {save_path}")
            agent.save_model(save_path)
            last_save_step = global_step

        if batch_episode_rewards:
             avg_ep_reward = np.mean(batch_episode_rewards)
             avg_ep_length = np.mean(batch_episode_lengths)
             logger.info(f"Batch Stats (Last {len(batch_episode_rewards)} episodes) - Avg Reward: {avg_ep_reward:.2f}, Avg Length: {avg_ep_length:.1f}")
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