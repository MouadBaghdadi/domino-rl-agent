import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from environment.utils import DEVICE
from agents.agentsbelief_network import BeliefNetwork
MAX_DOMINO_VALUE = 6  
HAND_SIZE = 7  
TOTAL_TILES = 28  #

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
        """
        Passe avant.
        obs: Tensor de l'observation. Doit avoir une dimension de séquence (batch, seq_len, features).
             Pour une seule observation: (1, 1, features).
        lstm_hidden: Tuple (h_n, c_n) de l'état caché LSTM précédent.
        """
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1) # (batch, features) -> (batch, 1, features)

        lstm_out, new_hidden = self.lstm(obs, lstm_hidden)

        last_lstm_out = lstm_out[:, -1, :]

        shared_features = self.shared_layer(last_lstm_out)

        action_logits = self.actor_head(shared_features)

        state_value = self.critic_head(shared_features)

        return action_logits, state_value, new_hidden

class PpoLstmAgent:
    def __init__(self, observation_dim, action_dim, config: Dict[str, Any]):
        self.config = config
        self.gamma = config.get('gamma', 0.99)
        self.lamda = config.get('lambda', 0.95) # Pour GAE
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.value_loss_coeff = config.get('value_loss_coeff', 0.5)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        lr = config.get('learning_rate', 3e-4)
        lstm_hidden_dim = config.get('lstm_hidden_dim', 128)
        shared_hidden_dim = config.get('shared_hidden_dim', 128)

        self.network = ActorCriticLSTM(observation_dim, action_dim, lstm_hidden_dim, shared_hidden_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # self.belief_network = BeliefNetwork(initial_opponent_hand_size) # Initialiser avec la taille de main adverse

        # Mémoire pour LSTM (état caché)
        self.lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.memory = [] # Devrait contenir (obs, action, reward, done, log_prob, value, next_obs)

    def reset(self):
        """Réinitialise l'état caché LSTM pour un nouvel épisode."""
        self.lstm_hidden = None
        # self.belief_network.reset()

    def _preprocess_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Convertit l'observation du dictionnaire Gym en tenseur pour le réseau."""
        my_hand = torch.tensor(obs['my_hand'], dtype=torch.float32)
        open_ends = torch.tensor(obs['open_ends'], dtype=torch.float32) / (MAX_DOMINO_VALUE + 1)
        opponent_hand_size = torch.tensor(obs['opponent_hand_size'], dtype=torch.float32) / HAND_SIZE 
        draw_pile_size = torch.tensor(obs['draw_pile_size'], dtype=torch.float32) / (TOTAL_TILES - 2*HAND_SIZE) 
        current_player = torch.tensor(obs['current_player'], dtype=torch.float32)

        processed_obs = torch.cat([
            my_hand,
            open_ends,
            opponent_hand_size,
            draw_pile_size,
            current_player
        ], dim=-1).unsqueeze(0) 

        # belief_probs = torch.tensor(self.belief_network.get_probabilities(), dtype=torch.float32).unsqueeze(0)
        # processed_obs = torch.cat([processed_obs, belief_probs], dim=-1)

        return processed_obs.to(DEVICE)


    def select_action(self, obs: Dict[str, Any], legal_action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Sélectionne une action en utilisant la politique actuelle."""
        processed_obs = self._preprocess_obs(obs)

        self.network.eval() 
        with torch.no_grad():
            action_logits, state_value, self.lstm_hidden = self.network(processed_obs, self.lstm_hidden)

        legal_mask_tensor = torch.tensor(legal_action_mask, dtype=torch.bool, device=DEVICE)
        masked_logits = torch.where(legal_mask_tensor, action_logits, torch.tensor(float('-inf'), device=DEVICE))

        action_probs = Categorical(logits=masked_logits)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob.item(), state_value.item() # action, log_prob, value

    def learn(self):
        """Updates the network using collected trajectory data with PPO algorithm."""
        if len(self.memory) == 0:
            return
        
        obs_list, action_list, reward_list, done_list, old_log_prob_list, old_value_list, next_obs_list = zip(*self.memory)
        
        obs_tensor = torch.cat([self._preprocess_obs(o) for o in obs_list], dim=0)
        action_tensor = torch.tensor(action_list, dtype=torch.long, device=DEVICE)
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=DEVICE)
        done_tensor = torch.tensor(done_list, dtype=torch.float32, device=DEVICE)
        old_log_prob_tensor = torch.tensor(old_log_prob_list, dtype=torch.float32, device=DEVICE)
        old_value_tensor = torch.tensor(old_value_list, dtype=torch.float32, device=DEVICE)
        
        returns = []
        advantages = []
        gae = 0
        next_value = 0  
        
        for t in reversed(range(len(reward_list))):
            if t == len(reward_list) - 1:
                if not done_list[t]:  
                    with torch.no_grad():
                        next_value = self.network(self._preprocess_obs(next_obs_list[t]), None)[1].item()
                next_non_terminal = 1.0 - done_list[t]
            else:
                next_non_terminal = 1.0 - done_list[t]
                next_value = old_value_list[t+1]
            
            delta = reward_list[t] + self.gamma * next_value * next_non_terminal - old_value_list[t]
            gae = delta + self.gamma * self.lamda * next_non_terminal * gae
            
            returns.insert(0, gae + old_value_list[t])
            advantages.insert(0, gae)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        self.network.train()
        
        dataset_size = len(obs_list)
        indices = np.arange(dataset_size)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = torch.cat([self._preprocess_obs(obs_list[i]) for i in batch_indices], dim=0)
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_prob_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                batch_size = len(batch_indices)
                batch_lstm_hidden = (
                    torch.zeros(1, batch_size, self.network.lstm_hidden_dim, device=DEVICE),
                    torch.zeros(1, batch_size, self.network.lstm_hidden_dim, device=DEVICE)
                )
                
                action_logits, values, _ = self.network(batch_obs, batch_lstm_hidden)
                values = values.squeeze(-1)
                
                dist = Categorical(logits=action_logits)
                
                new_log_probs = dist.log_prob(batch_actions)
                
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy
 
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
    
        self.memory.clear()
        
        self.network.eval()

    def store_transition(self, obs, action, reward, done, log_prob, value, next_obs):
         """Stocke une transition dans la mémoire."""
         self.memory.append((obs, action, reward, done, log_prob, value, next_obs))

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        """Load model from a file path or directly from a state dictionary.
        
        Args:
            path: Either a string (file path) or a state_dict (OrderedDict)
        """
        if isinstance(path, str):
            self.network.load_state_dict(torch.load(path, map_location=DEVICE))
        else:
            self.network.load_state_dict(path)