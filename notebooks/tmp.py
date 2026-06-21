import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define weights and biases for all gates
        # Forget Gate: W_f, U_f, b_f
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        
        # Input Gate: W_i, U_i, b_i
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        
        # Candidate Cell State: W_c, U_c, b_c
        self.W_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output Gate: W_o, U_o, b_o
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        batch_size, seq_len, _ = x.size()
        
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = init_states
        
        hidden_states = []
        cell_states = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Forget Gate
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            
            # Input Gate
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            
            # Candidate Cell State
            c_tilde_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)
            
            # New Cell State
            c_t = f_t * c_t + i_t * c_tilde_t
            
            # Output Gate
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            
            # New Hidden State
            h_t = o_t * torch.tanh(c_t)
            
            hidden_states.append(h_t)
            cell_states.append(c_t)
            
        hidden_states = torch.stack(hidden_states, dim=1)
        
        return hidden_states, (h_t, c_t)