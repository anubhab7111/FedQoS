import torch
import torch.nn as nn

class LSTM(nn.Module):
    """LSTM Model for multi-output QoS prediction."""
    def __init__(self, input_size=5, num_target_features=2, future_window_len=5, 
                 hidden_size=1024, num_layers=3, dropout=0.3, bidirectional=False):
        super(LSTM, self).__init__()
        self.num_target_features = num_target_features
        self.future_window_len = future_window_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 128),
            nn.ReLU(),
            nn.Linear(128, future_window_len * num_target_features)
        )

    def forward(self, x):
        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # take last time step
        out = self.fc(out)
        # Reshape to (batch, future_window_len, num_target_features)
        out = out.view(x.size(0), self.future_window_len, self.num_target_features)
        return out

def build_model(config):
    """Build the LSTM from a loaded config. Every entry point must use this so architectures match."""
    return LSTM(
        input_size=config["model"]["input_size"],
        num_target_features=len(config["data"]["target_cols"]),
        future_window_len=config["data"]["future_win_len"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"]
    )
