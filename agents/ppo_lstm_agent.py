from venv import logger
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from environment.utils import DEVICE, MAX_DOMINO_VALUE, HAND_SIZE, TOTAL_TILES, ALL_DOMINOS
from .agentsbelief_network import BeliefNetwork

class PPOMemory:
    def __init__(self, batch_size):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size 

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_data(self):
        obs_tensor = torch.tensor(np.array(self.obs), dtype=torch.float32).to(DEVICE)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long).to(DEVICE)
        log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32).to(DEVICE)
        values_tensor = torch.tensor(self.values, dtype=torch.float32).to(DEVICE)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32).to(DEVICE)
        dones_tensor = torch.tensor(self.dones, dtype=torch.bool).to(DEVICE)
        return obs_tensor, actions_tensor, log_probs_tensor, values_tensor, rewards_tensor, dones_tensor

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.dones)

class ActorCriticLSTM(nn.Module):
    def __init__(self, observation_dim, action_dim, lstm_hidden_dim=128, shared_hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(observation_dim, lstm_hidden_dim, batch_first=True)
        self.shared_layer = nn.Sequential(
            nn.Linear(lstm_hidden_dim, shared_hidden_dim),
            nn.ReLU() 
        )
        self.actor_head = nn.Linear(shared_hidden_dim, action_dim)
        self.critic_head = nn.Linear(shared_hidden_dim, 1)

    def forward(self, obs: torch.Tensor, lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1) 
        if lstm_hidden is None:
             batch_size = obs.size(0)
             h_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(obs.device)
             c_0 = torch.zeros(1, batch_size, self.lstm_hidden_dim).to(obs.device)
             lstm_hidden = (h_0, c_0)

        lstm_out, new_hidden = self.lstm(obs, lstm_hidden)
        last_lstm_out = lstm_out[:, -1, :] 
        shared_features = self.shared_layer(last_lstm_out)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value, new_hidden

class PpoLstmAgent:
    def __init__(self, action_dim: int, config: Dict[str, Any]):
        self.config = config
        self.gamma = config.get('gamma', 0.99)
        self.lamda = config.get('lambda', 0.95) 
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.value_loss_coeff = config.get('value_loss_coeff', 0.5)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        lr = config.get('learning_rate', 3e-4)
        self.max_grad_norm = config.get('max_grad_norm', 0.5) 
        lstm_hidden_dim = config.get('lstm_hidden_dim', 128)
        shared_hidden_dim = config.get('shared_hidden_dim', 128)

        self.observation_dim = self._calculate_observation_dim(config) 
        self.action_dim = action_dim

        self.network = ActorCriticLSTM(self.observation_dim, self.action_dim, lstm_hidden_dim, shared_hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.use_belief_network = config.get('use_belief_network', True) 
        if self.use_belief_network:
            self.belief_network = BeliefNetwork()
        else:
            self.belief_network = None

        self.lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.memory = PPOMemory(batch_size=self.batch_size) 

        self.last_train_metrics = {}

    def update_hyperparameters(self, learning_rate: Optional[float] = None, entropy_coeff: Optional[float] = None):
        """Met à jour les hyperparamètres de l'agent (ex: pour le curriculum)."""
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            logger.info(f"Learning rate updated to: {learning_rate}")

        if entropy_coeff is not None:
            self.entropy_coeff = entropy_coeff
            logger.info(f"Entropy coefficient updated to: {entropy_coeff}")


    def _calculate_observation_dim(self, config: Dict[str, Any]) -> int:
        """ Calcule la dimension de l'observation en fonction de la configuration. """
        hand_dim = len(ALL_DOMINOS)
        open_ends_dim = 2
        scalar_dim = 2
        board_history_dim = MAX_DOMINO_VALUE + 1
        belief_dim = len(ALL_DOMINOS) if config.get('use_belief_network', True) else 0

        total_dim = hand_dim + open_ends_dim + scalar_dim + board_history_dim + belief_dim
        return total_dim


    def reset(self):
        """Réinitialise l'état caché LSTM et le Belief Network pour un nouvel épisode."""
        if self.lstm_hidden is not None:
            self.lstm_hidden = (self.lstm_hidden[0].detach(), self.lstm_hidden[1].detach())
        else:
             self.lstm_hidden = None 

        if self.belief_network:
            self.belief_network.reset()


    def _preprocess_obs(self, obs: Dict[str, Any], info: Dict[str, Any]) -> torch.Tensor:
        """
        Convertit l'observation du dictionnaire Gym et les infos en tenseur pour le réseau.
        C'est une étape CRUCIALE et dépend fortement de ce que contient l'observation de l'env.
        """
        processed_features = []

        my_hand_encoding = torch.tensor(obs['my_hand'], dtype=torch.float32)
        processed_features.append(my_hand_encoding)

        # obs['open_ends'] est supposé être np.array([end1, end2]), avec MAX_DOMINO_VALUE+1 si pas d'extrémité.
        open_ends_normalized = torch.tensor(obs['open_ends'], dtype=torch.float32) / (MAX_DOMINO_VALUE + 1.0)
        processed_features.append(open_ends_normalized)

        opponent_hand_size_norm = torch.tensor([obs['opponent_hand_size'].item()], dtype=torch.float32) / HAND_SIZE
        draw_pile_size_norm = torch.tensor([obs['draw_pile_size'].item()], dtype=torch.float32) / max(1, TOTAL_TILES - 2*HAND_SIZE)
        processed_features.append(opponent_hand_size_norm)
        processed_features.append(draw_pile_size_norm)

        board_value_counts = np.zeros(MAX_DOMINO_VALUE + 1)
        if 'board' in info: 
             for tile in info['board']:
                 s1, s2 = tile.get_sides()
                 board_value_counts[s1] += 1
                 if not tile.is_double():
                     board_value_counts[s2] += 1
        max_count_per_value = (MAX_DOMINO_VALUE + 1) + 1 # (0-6) + double = 8
        board_value_counts_norm = torch.tensor(board_value_counts, dtype=torch.float32) / max_count_per_value
        processed_features.append(board_value_counts_norm)


        if self.belief_network:
            belief_probs = torch.tensor(self.belief_network.get_probabilities(), dtype=torch.float32)
            processed_features.append(belief_probs)


        final_obs_tensor = torch.cat(processed_features, dim=-1)

        final_obs_tensor = final_obs_tensor.unsqueeze(0)

        expected_dim = self._calculate_observation_dim(self.config)
        if final_obs_tensor.shape[-1] != expected_dim:
             print(f"WARN: Dimension d'observation prétraitée ({final_obs_tensor.shape[-1]}) != attendue ({expected_dim})")

        return final_obs_tensor.to(DEVICE)


    def select_action(self, obs: Dict[str, Any], info: Dict[str, Any], legal_action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Sélectionne une action en utilisant la politique actuelle."""
        processed_obs = self._preprocess_obs(obs, info) 

        self.network.eval() 
        with torch.no_grad():
            action_logits, state_value, new_hidden = self.network(processed_obs, self.lstm_hidden)
            self.lstm_hidden = (new_hidden[0].detach(), new_hidden[1].detach())


        legal_mask_tensor = torch.tensor(legal_action_mask, dtype=torch.bool, device=action_logits.device)
        if legal_mask_tensor.shape[0] != action_logits.shape[-1]:
             print(f"ERROR: Discrepance taille masque ({legal_mask_tensor.shape[0]}) et logits ({action_logits.shape[-1]})!")
             legal_indices = np.where(legal_action_mask)[0]
             action = np.random.choice(legal_indices) if len(legal_indices) > 0 else 0 
             log_prob = np.log(1.0 / len(legal_indices)) if len(legal_indices) > 0 else -np.inf
             return action, log_prob, state_value.item() 

        masked_logits = torch.where(legal_mask_tensor, action_logits, torch.tensor(float('-inf'), device=action_logits.device))

        if torch.all(torch.isinf(masked_logits)):
             print("ERROR: Toutes les actions sont masquées comme illégales ou logits sont infinis!")
             legal_indices = np.where(legal_action_mask)[0]
             action = np.random.choice(legal_indices) if len(legal_indices) > 0 else 0
             log_prob = np.log(1.0 / len(legal_indices)) if len(legal_indices) > 0 else -np.inf
             return action, log_prob, state_value.item()


        try:
            action_probs = Categorical(logits=masked_logits) 
            action = action_probs.sample()
            log_prob = action_probs.log_prob(action)
        except ValueError as e:
             print(f"ERROR: Erreur lors de la création de Categorical ou de l'échantillonnage: {e}")
             print(f"Masked Logits: {masked_logits}")
             # Fallback
             legal_indices = np.where(legal_action_mask)[0]
             action = np.random.choice(legal_indices) if len(legal_indices) > 0 else 0
             log_prob = np.log(1.0 / len(legal_indices)) if len(legal_indices) > 0 else -np.inf
             return action, log_prob, state_value.item()


        return action.item(), log_prob.item(), state_value.item() 


    def store_transition(self, obs_tensor: torch.Tensor, action: int, reward: float, done: bool, log_prob: float, value: float):
        """Stocke les éléments d'une transition dans la mémoire tampon (self.memory)."""
        if not isinstance(obs_tensor, np.ndarray):
             obs_np = obs_tensor.squeeze(0).cpu().numpy()
        else:
             obs_np = obs_tensor

        self.memory.store(obs_np, action, log_prob, value, reward, done)

    def clear_memory(self):
        """Vide la mémoire tampon (self.memory) après une mise à jour."""
        self.memory.clear()

    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Calcule les avantages en utilisant Generalized Advantage Estimation (GAE)
        et les retours (targets pour le critique).

        Args:
            rewards: Liste des récompenses reçues à chaque pas.
            values: Liste des valeurs estimées par le critique pour chaque état.
            dones: Liste des indicateurs de fin d'épisode.
            last_value: Valeur estimée de l'état après la dernière action de la trajectoire
                        (utilisé pour le bootstrap si la trajectoire ne s'est pas terminée).

        Returns:
            Tuple[List[float], List[float]]: Liste des avantages, Liste des retours (value targets).
        """
        advantages = []
        returns = []
        gae = 0.0
        next_value = last_value

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t] 
            delta = rewards[t] + self.gamma * next_value * mask - values[t]

            gae = delta + self.gamma * self.lamda * gae * mask

            return_t = gae + values[t]

            advantages.insert(0, gae) 
            returns.insert(0, return_t)

            next_value = values[t]

        return advantages, returns

    def learn(self):
        """
        Met à jour les réseaux Acteur et Critique en utilisant l'algorithme PPO
        et les données collectées dans `trajectory_data`.

        REMARQUE IMPORTANTE: Implémenter PPO et gérer correctement les états LSTM
        pendant l'entraînement par batch est complexe. L'utilisation de bibliothèques
        robustes comme Stable-Baselines3 est FORTEMENT recommandée si l'objectif
        principal n'est pas de réimplémenter PPO vous-même.

        Lien vers Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

        Si vous souhaitez implémenter vous-même, voici les étapes clés :
        """

        obs_tensor, actions_tensor, old_log_probs_tensor, values_tensor, rewards_tensor, dones_tensor = self.memory.get_data()

        if len(obs_tensor) == 0:
            logger.warning("Learn() appelé mais pas de données dans le buffer. Saut de l'apprentissage.")
            self.last_train_metrics = {} 
            return

        with torch.no_grad():
            last_obs = obs_tensor[-1].unsqueeze(0) #
            _, last_value_tensor, _ = self.network(last_obs, self.lstm_hidden) 
            last_value = last_value_tensor.item() if not dones_tensor[-1] else 0.0

        advantages_list, returns_list = self._compute_gae(
            rewards_tensor.cpu().numpy().tolist(),  
            values_tensor.cpu().numpy().tolist(),
            dones_tensor.cpu().numpy().tolist(),
            last_value
        )
        advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(DEVICE)
        returns_tensor = torch.tensor(returns_list, dtype=torch.float32).to(DEVICE)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        num_samples = len(obs_tensor)
        indices = np.arange(num_samples)

        self.network.train()

        for epoch in range(self.epochs):
            np.random.shuffle(indices) 

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                current_logits, current_values, _ = self.network(batch_obs, lstm_hidden=None) 
                current_values = current_values.squeeze(-1)

                dist = Categorical(logits=current_logits)
                current_log_probs = dist.log_prob(batch_actions)

                entropy = dist.entropy().mean()

                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() 

                values_pred_clipped = values_tensor[batch_indices] + torch.clamp(current_values - values_tensor[batch_indices], -self.clip_epsilon, self.clip_epsilon)
                vf_loss1 = (current_values - batch_returns).pow(2)
                vf_loss2 = (values_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()


                loss = actor_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.last_train_metrics = {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'mean_advantage': advantages_tensor.mean().item(),
            'mean_return': returns_tensor.mean().item()
        }


    def save_model(self, path):
        """Sauvegarde l'état du réseau et potentiellement de l'optimizer."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Modèle sauvegardé sur {path}")

    def load_model(self, path):
        """Charge l'état du réseau et de l'optimizer."""
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.network.to(DEVICE) 
            print(f"Modèle chargé depuis {path}")
        except FileNotFoundError:
            print(f"ERREUR: Fichier modèle non trouvé à {path}")
            raise
        except Exception as e:
             print(f"ERREUR lors du chargement du modèle depuis {path}: {e}")
             raise