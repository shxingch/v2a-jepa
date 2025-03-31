# src/models/dynamics_model.py
import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    """预测给定当前状态和动作后的下一状态表示"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        model_type="mlp",
        num_layers=2,
        use_layer_norm=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = model_type
        
        if model_type == "mlp":
            # 简单MLP动态模型
            layers = [nn.Linear(state_dim + action_dim, hidden_dim)]
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                
            layers.append(nn.Linear(hidden_dim, state_dim))
            self.dynamics = nn.Sequential(*layers)
            
        elif model_type == "transformer":
            # Transformer动态模型 - 更复杂但可能更强大
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.action_proj = nn.Linear(action_dim, state_dim)
            self.output_proj = nn.Linear(state_dim, state_dim)
            
    def forward(self, state, action):
        """
        Args:
            state: 编码后的状态表示 [B, N, D]
            action: 动作表示 [B, A]
        Returns:
            predicted_next_state: 预测的下一状态表示 [B, N, D]
        """
        batch_size, num_tokens, state_dim = state.shape
        
        if self.model_type == "mlp":
            # [B, A] -> [B, N, A] 广播动作到每个状态token
            action_expanded = action.unsqueeze(1).expand(-1, num_tokens, -1)
            
            # 将状态和动作拼接
            state_action = torch.cat([state, action_expanded], dim=-1)
            
            # 对每个token应用动态模型
            flat_state_action = state_action.reshape(-1, state_dim + self.action_dim)
            flat_next_state = self.dynamics(flat_state_action)
            next_state = flat_next_state.reshape(batch_size, num_tokens, state_dim)
            
        elif self.model_type == "transformer":
            # 将动作投影到状态空间
            action_emb = self.action_proj(action).unsqueeze(1)  # [B, 1, D]
            
            # 拼接动作和状态
            state_action = torch.cat([action_emb, state], dim=1)  # [B, N+1, D]
            
            # 应用Transformer编码器
            next_state_action = self.transformer(state_action)
            
            # 提取预测的下一状态（去除动作token）
            next_state = self.output_proj(next_state_action[:, 1:, :])
            
        return next_state