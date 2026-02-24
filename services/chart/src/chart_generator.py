import torch
import numpy as np

class ChartGenerator:
    def __init__(self, width: int = 1000, height: int = 64):
        self.width = width
        self.height = height
        self.y_coords = np.arange(height)[:, None]  # Shape (64, 1)

    def generate_chart(self, open_prices, high_prices, low_prices, close_prices, volume) -> torch.Tensor:
        o = np.asarray(open_prices)
        h = np.asarray(high_prices)
        l = np.asarray(low_prices)
        c = np.asarray(close_prices)
        
        min_p = l.min()
        max_p = h.max()
        range_p = max_p - min_p + 1e-8
        
        def scale(arr):
            return np.round((arr - min_p) / range_p * (self.height - 1)).astype(int)
            
        o_s, h_s, l_s, c_s = scale(o), scale(h), scale(l), scale(c)
        
        wick_mask = (self.y_coords >= l_s) & (self.y_coords <= h_s)
        
        body_min = np.minimum(o_s, c_s)
        body_max = np.maximum(o_s, c_s)
        body_mask = (self.y_coords >= body_min) & (self.y_coords <= body_max)
        
        up_mask = c >= o
        down_mask = ~up_mask
        
        img = np.zeros((self.height, self.width), dtype=np.float32)
        
        img[wick_mask] = 0.5
        
        body_up_mask = body_mask & up_mask
        img[body_up_mask] = 1.0
        
        body_down_mask = body_mask & down_mask
        img[body_down_mask] = 0.2
        
        return torch.from_numpy(img).unsqueeze(0)
