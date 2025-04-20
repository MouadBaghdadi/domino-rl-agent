import torch
import torch.nn as nn

class OpponentModel(nn.Module):
    """Modélisation de la stratégie de l'adversaire avec LSTM"""
    def __init__(self, input_dim=28+40):  # board(20*2) + hidden_state(28)
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.policy_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 28))
        
        self.value_net = nn.Linear(128, 1)

    def forward(self, seq_observations):
        # seq_observations: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(seq_observations)
        last_out = lstm_out[:, -1, :]
        
        policy_logits = self.policy_net(last_out)
        value = self.value_net(last_out)
        
        return policy_logits, value

    def predict_action(self, obs_history):
        """Prédit la prochaine action de l'adversaire"""
        with torch.no_grad():
            logits, _ = self(obs_history)
            probs = torch.softmax(logits, dim=-1)
            probs *= obs_history[-1, :28]  # Mask des actions valides
            return torch.argmax(probs).item()