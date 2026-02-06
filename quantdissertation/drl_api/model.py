import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ExplainableTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        self.seq_len = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]
        self.d_model = 128
        
        self.linear_in = nn.Linear(self.input_dim, self.d_model)
        # Note: average_attn_weights=False is key for explainability
        self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.linear_out = nn.Linear(self.seq_len * self.d_model, features_dim)
        self.act = nn.Tanh()
        
        self.latest_attn_weights = None

    def forward(self, observations):
        x = self.linear_in(observations)
        attn_output, attn_weights = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        self.latest_attn_weights = attn_weights.detach().cpu().numpy()
        x = self.norm1(x + attn_output)
        x = x.flatten(start_dim=1)
        return self.act(self.linear_out(x))
