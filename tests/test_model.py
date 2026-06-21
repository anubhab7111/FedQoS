import torch
from src.models.lstm import LSTM

def test_lstm_forward():
    batch_size = 8
    window_len = 15
    input_size = 5
    future_win_len = 5
    num_target_features = 2
    
    model = LSTM(
        input_size=input_size,
        num_target_features=num_target_features,
        future_window_len=future_win_len,
        hidden_size=64,
        num_layers=2
    )
    
    x = torch.randn(batch_size, window_len, input_size)
    y = model(x)
    
    assert y.shape == (batch_size, future_win_len, num_target_features)
    print("Test passed: LSTM forward shape is correct.")

if __name__ == "__main__":
    test_lstm_forward()
