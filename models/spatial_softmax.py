import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Biến đổi Feature Map (B, C, H, W) thành Feature Vector (B, C*2) chứa tọa độ (x, y).
    Giúp model giữ được thông tin vị trí không gian của vật thể.
    """
    def __init__(self, num_rows, num_cols, temperature=None):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else temperature

        # Tạo lưới tọa độ (pos_x, pos_y) chuẩn hóa về [-1, 1]
        x_map = torch.linspace(-1, 1, num_cols)
        y_map = torch.linspace(-1, 1, num_rows)
        
        # Tạo lưới tọa độ cho cả batch
        self.register_buffer('x_grid', x_map.view(1, 1, 1, -1)) # (1, 1, 1, W)
        self.register_buffer('y_grid', y_map.view(1, 1, -1, 1)) # (1, 1, H, 1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C*2)
        """
        n, c, h, w = x.size()
        
        # 1. Tính Softmax trên không gian (H, W) để tìm vùng "chú ý" nhất
        # x * temperature để điều chỉnh độ nhọn của phân phối
        x = x * self.temperature
        x = x.view(n, c, -1) # (B, C, H*W)
        softmax_attention = F.softmax(x, dim=2) # (B, C, H*W) - xác suất tại mỗi điểm ảnh
        
        # Reshape lại thành map
        softmax_attention = softmax_attention.view(n, c, h, w)
        
        # 2. Tính kỳ vọng tọa độ (Expected Coordinate)
        # Tọa độ X trung bình = Sum(xác suất * giá trị trục x)
        expected_x = torch.sum(softmax_attention * self.x_grid, dim=[2, 3]) # (B, C)
        expected_y = torch.sum(softmax_attention * self.y_grid, dim=[2, 3]) # (B, C)
        
        # 3. Ghép lại thành vector đặc trưng (x1, y1, x2, y2, ...)
        expected_xy = torch.cat((expected_x, expected_y), dim=1) # (B, C*2)
        
        return expected_xy