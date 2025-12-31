import torch 
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])  # Global average pooling (reduce overfiting when training data is small)
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)
    
class StateEncoder(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.ln(x)
        return x  
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)  # (B, T, d_word)
        _, h_n = self.gru(x)  # h_n: (1, B, d_model)
        h_n = h_n.squeeze(0)  # (B, d_model)
        h_n = self.ln(h_n)
        return h_n  
