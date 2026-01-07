import torch 
import torch.nn as nn
import torch.nn.functional as F
from .spatial_softmax import SpatialSoftmax

class ImageEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # Input: 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2) # -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # -> 16x16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) # -> 8x8 (Giảm channel chút cho nhẹ)
        
        # Spatial Softmax thay cho Global Average Pooling
        # Input feature map cuối cùng kích thước 8x8 (nếu ảnh gốc 64x64)
        self.spatial_softmax = SpatialSoftmax(num_rows=8, num_cols=8)
        
        # Sau Spatial Softmax, output dimension là Channel * 2 (tọa độ x, y cho mỗi channel)
        # Conv3 có 64 channel -> 64 * 2 = 128 features
        self.flatten_dim = 64 * 2 
        
        self.proj = nn.Linear(self.flatten_dim, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 3, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # (B, 64, 8, 8)
        
        # --- PHẦN QUAN TRỌNG NHẤT ---
        # Thay vì x.mean(), ta dùng Spatial Softmax
        x = self.spatial_softmax(x) # (B, 128) - Chứa tọa độ không gian
        # -----------------------------
        
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
