import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .belief_network import BeliefNetwork
from .opponent_model import OpponentModel
from environment.domino_env import DominoEnv
from .rule_based_bots import GreedyBot, RandomBot, DefensiveBot
import torch.nn.functional as F

import numpy as np
from torch.utils.data import Dataset, DataLoader

class PPODataset(Dataset):
    """Dataset pour stocker les trajectoires d'entraînement"""
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class PolicyNetwork(nn.Module):
    """Réseau principal avec mécanisme d'attention pour partial observability"""
    def __init__(self, belief_net, opponent_model):
        super().__init__()
        
        self.belief_net = belief_net
        self.opponent_model = opponent_model
        
        # Module d'attention
        self.query = nn.Linear(128, 64)
        self.key = nn.Linear(64, 64)
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),  # Concaténation des features
            nn.ReLU(),
            nn.Linear(256, 128))
        
        self.actor = nn.Linear(128, 28)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs, hidden):
        # Étape 1: Estimation de la croyance
        belief_probs, new_hidden = self.belief_net(obs, hidden)
        
        # Étape 2: Prédiction de l'adversaire
        opponent_logits = self.opponent_model(obs["board"])
        
        # Étape 3: Fusion par attention
        query = self.query(belief_probs)
        key = self.key(opponent_logits)
        attention = torch.softmax(torch.matmul(query, key.T), dim=-1)
        context = torch.matmul(attention, opponent_logits)
        
        # Étape 4: Décision finale
        fused = torch.cat([belief_probs, context], dim=-1)
        x = self.fusion(fused)
        
        return torch.softmax(self.actor(x), dim=-1), self.critic(x), new_hidden

class PPOTrainer:
    """Implémentation complète de PPO avec gestion de la partial observability"""
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.win_rate = 0.0
        
        # Initialisation des modèles
        self.belief_net = BeliefNetwork().to(device)
        self.opponent_model = OpponentModel().to(device)
        self.policy = PolicyNetwork(self.belief_net, self.opponent_model).to(device)
        
        # Optimiseurs
        self.optimizer = optim.Adam(
            self.policy.parameters(),  # Uniquement les paramètres du policy network
            lr=3e-4,
            weight_decay=1e-5
        )
        
        # Hyperparamètres
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.batch_size = 4096

    def train(self, num_episodes):
        for episode in range(num_episodes):
            batch = self._collect_experience()
            self.update_policy(batch)
            
            # Logging et sauvegarde
            if (episode+1) % 100 == 0:
                self.evaluate()
                self._save_checkpoint(episode+1)

    def get_action(self, obs, training=True):
        """
        Obtient une action à partir de l'observation actuelle
        Args:
            obs (dict): Observation de l'environnement
            training (bool): Si True, utilise l'exploration (sinon mode évaluation)
        """
        # Conversion en tenseur
        state_tensor = {
            k: torch.FloatTensor(v).unsqueeze(0).to(self.device)  # Ajoute une dimension batch
            for k, v in obs.items()
        }
        
        with torch.no_grad():
            action_probs, _, _ = self.policy(state_tensor, None)
            
        # Application du masque des actions valides
        valid_mask = torch.FloatTensor(obs['valid_actions']).to(self.device)
        masked_probs = action_probs * valid_mask
        
        # Normalisation
        masked_probs /= masked_probs.sum() + 1e-8
        
        # Choix de l'action
        if training:
            dist = Categorical(masked_probs)
            action = dist.sample().item()
        else:
            action = torch.argmax(masked_probs).item()
            
        return action


    def _collect_experience(self):
        """Collecte des trajectoires avec gestion de la mémoire LSTM"""
        batch = []
        hidden_states = []
        episode_rewards = []
        
        # Initialisation de l'état caché LSTM
        lstm_hidden = (
            torch.zeros(1, 1, 64).to(self.device),
            torch.zeros(1, 1, 64).to(self.device)
        )
        
        # Collecte de N trajectoires
        for _ in range(self.batch_size):
            obs = self.env.reset()
            done = False
            episode = {
                'states': [], 'actions': [], 
                'rewards': [], 'dones': [],
                'values': [], 'logprobs': [],
                'hidden': []
            }
            
            while not done:
                # Conversion en tenseur
                state_tensor = {
                    k: torch.FloatTensor(v).to(self.device) 
                    for k, v in obs.items()
                }
                
                # Passage dans le réseau
                with torch.no_grad():
                    action_probs, value, lstm_hidden = self.policy(
                        state_tensor, lstm_hidden
                    )
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Exécution de l'action
                next_obs, reward, done, _ = self.env.step(action.cpu().numpy())
                
                # Stockage des données
                episode['states'].append(obs)
                episode['actions'].append(action)
                episode['rewards'].append(reward)
                episode['dones'].append(done)
                episode['values'].append(value)
                episode['logprobs'].append(log_prob)
                episode['hidden'].append(lstm_hidden)
                
                # Mise à jour de la croyance si l'adversaire joue
                if not done:
                    self.belief_net.update_belief(
                        self.env.id_to_domino(next_obs['last_action'])
                    )
                
                obs = next_obs
            
            # Calcul des avantages pour l'épisode
            episode_rewards.append(sum(episode['rewards']))
            batch.append(episode)
        
        print(f"Reward moyen batch: {np.mean(episode_rewards):.2f}")
        return batch

    def _compute_advantages(self, rewards, values, masks, gamma=0.99, tau=0.95):
        """Implémentation de Generalized Advantage Estimation (GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Calcul inverse dans le temps
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * masks[t] - values[t]
            gae = delta + gamma * tau * masks[t] * gae
            advantages[t] = gae
        
        # Normalisation
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update_policy(self, batch):
        """Mise à jour du policy avec clipping PPO"""
        dataset = PPODataset(batch)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(4):  # Nombre d'epochs par batch
            for episode in loader:
                # Conversion en tenseurs
                states = episode['states']
                old_logprobs = torch.stack(episode['logprobs']).to(self.device)
                actions = torch.stack(episode['actions']).to(self.device)
                rewards = torch.FloatTensor(episode['rewards']).to(self.device)
                dones = torch.FloatTensor(episode['dones']).to(self.device)
                values = torch.stack(episode['values']).to(self.device)
                hidden = episode['hidden']
                
                # Calcul des avantages
                advantages = self._compute_advantages(rewards, values, 1 - dones)
                
                # Calcul des nouvelles probabilités
                new_logprobs = []
                new_values = []
                for state, h in zip(states, hidden):
                    state_tensor = {k: torch.FloatTensor(v).to(self.device) for k, v in state.items()}
                    probs, val, _ = self.policy(state_tensor, h)
                    dist = Categorical(probs)
                    new_logprobs.append(dist.log_prob(actions))
                    new_values.append(val)
                
                new_logprobs = torch.stack(new_logprobs)
                new_values = torch.stack(new_values)
                
                # Calcul des ratios
                ratios = torch.exp(new_logprobs - old_logprobs.detach())
                
                # Loss PPO
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Loss critic
                critic_loss = F.mse_loss(new_values, rewards + self.gamma * (1 - dones) * new_values[1:])
                
                # Entropie
                entropy = Categorical(probs).entropy().mean()
                
                # Loss totale
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Mise à jour
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def evaluate(self, num_games=100):
        """Évaluation contre différents bots rule-based"""        
        bots = [GreedyBot(), RandomBot(), DefensiveBot()]
        win_rates = {bot.__class__.__name__: [] for bot in bots}
        
        for bot in bots:
            wins = 0
            for _ in range(num_games):
                obs = self.env.reset()
                done = False
                current_player = 0
                
                while not done:
                    if current_player == 0:  # Notre agent
                        action = self._get_optimal_action(obs)
                    else:  # Bot
                        action = bot.act(obs)
                    
                    obs, reward, done, _ = self.env.step(action)
                    current_player = 1 - current_player
                
                if self.env.winner == 0:
                    wins += 1
            
            win_rate = wins / num_games
            win_rates[bot.__class__.__name__] = win_rate
            print(f"Win rate vs {bot.__class__.__name__}: {win_rate:.2%}")
        
        return win_rates

    def _get_optimal_action(self, obs):
        """Action optimale sans exploration"""
        state_tensor = {k: torch.FloatTensor(v).to(self.device) for k, v in obs.items()}
        with torch.no_grad():
            probs, _, _ = self.policy(state_tensor, None)
            valid_actions = torch.FloatTensor(obs['valid_actions']).to(self.device)
            masked_probs = probs * valid_actions
            return torch.argmax(masked_probs).item()

    def _save_checkpoint(self, episode):
        """Sauvegarde des modèles"""
        torch.save({
            'policy': self.policy.state_dict(),
            'belief_net': self.belief_net.state_dict(),
            'opponent_model': self.opponent_model.state_dict(),
        }, f"models/checkpoint_{episode}.pt")