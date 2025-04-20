import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefNetwork(nn.Module):
    """Réseau de croyance bayésien pour estimer la main de l'adversaire"""
    def __init__(self, input_dim=28+40+28):  # hand(28) + board(20*2) + valid_actions(28)
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64))
        
        self.lstm = nn.LSTM(64, 64)
        
        self.prob_head = nn.Sequential(
            nn.Linear(64, 28),
            nn.Sigmoid())  # Probabilité que chaque domino soit chez l'adversaire

    def forward(self, obs, hidden=None):
        # obs: dict de l'observation
        # Aplatir le board en conservant la dimension batch
        board_flat = obs["board"].view(obs["board"].size(0), -1)  # (batch, 40)
        
        # Concaténation correcte
        x = torch.cat([
            obs["hand"],        # shape: (batch, 28)
            board_flat,         # shape: (batch, 40)
            obs["valid_actions"] # shape: (batch, 28)
        ], dim=1)  # Résultat: (batch, 28+40+28=96)
        
        x = self.encoder(x)
        x, new_hidden = self.lstm(x.unsqueeze(0), hidden)
        probs = self.prob_head(x.squeeze(0))
        
        return probs, new_hidden

    def update_belief(self, played_domino_id):
        """Mise à jour bayésienne après qu'un domino soit joué"""
        with torch.no_grad():
            self.probs[:, played_domino_id] = 0.0  # Le domino ne peut plus être chez l'adversaire
            self.probs = F.normalize(self.probs, p=1, dim=1)  # Re-normalisation