import torch
import torch.nn as nn
from torchvision.models import resnet18


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv):
        '''
        q: [B, N_q, C]
        kv: [B, N_kv, C]
        '''
        B, N_q, C = q.shape
        _, N_kv, _ = kv.shape

        q = self.q(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FusionMLP(nn.Module):
    """MLP-based temporal attention fusion."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # scalar attention score
        )

    def forward(self, x):
        """
        x: (B, T, C)
        return: (B, C)
        """
        B, T, C = x.shape
        att_scores = self.mlp(x.view(B * T, C))     # (B*T, 1)
        att_scores = att_scores.view(B, T, 1)       # (B, T, 1)
        att_weights = torch.softmax(att_scores, dim=1)  # (B, T, 1)
        fused = torch.sum(x * att_weights, dim=1)   # (B, C)
        return fused


class FusionTransformer(nn.Module):
    """Lightweight Transformer-based temporal fusion."""
    def __init__(self, embed_dim=128, num_heads=4, num_layers=1, dropout=0.1, droppath=0.0):
        super().__init__()
        self.droppath = droppath
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True   # input shape: (B, T, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B, T, C)
        return: (B, C)
        """
        out = self.encoder(x)        # (B, T, C)
        fused = out.mean(dim=1)      # mean pooling over the temporal dimension
        return fused




class TemporalAttention(nn.Module):
    """
    Temporal self-attention for features shaped as (B, T, C', Hf, Wf).
    The spatial dimensions are flattened for attention and restored afterwards.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim

        # QKV projections.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional positional encoding.
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 16, dim) * 0.02)  # supports up to 16 time steps

    def forward(self, x):
        """
        x: (B, T, C', Hf, Wf)
        return: (B, T, C', Hf, Wf)
        """
        B, T, C, Hf, Wf = x.shape

        # Flatten spatial dimensions: (B, T, C', Hf, Wf) -> (B, T, C'*Hf*Wf).
        x_flat = x.reshape(B, T, C * Hf * Wf)  # (B, T, D) where D = C'*Hf*Wf

        # Add positional encoding when enabled.
        if self.use_pos_encoding and T <= self.pos_encoding.size(1):
            x_flat = x_flat + self.pos_encoding[:, :T, :C * Hf * Wf]

        # Self-attention.
        # x_flat: (B, T, D)
        qkv = self.qkv(x_flat)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, (C * Hf * Wf) // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, T, head_dim)

        # Compute attention weights.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention.
        x = (attn @ v).transpose(1, 2).reshape(B, T, C * Hf * Wf)  # (B, T, D)

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore spatial structure: (B, T, C'*Hf*Wf) -> (B, T, C', Hf, Wf).
        x_out = x.reshape(B, T, C, Hf, Wf)

        return x_out

class DCDPDynamicsExtractor(nn.Module):
    """Visual dynamics extractor used by DCDP training and evaluation."""

    def __init__(self, reduction=4, freeze_backbone=True, fusion_type="transformer", 
        fusion_transformer_config=None, cross_attn_drop=0.5, 
        cross_proj_drop=0.5, temp_attn_drop=0.5, temp_proj_drop=0.5,
        pretrained_backbone=False):

        super().__init__()
        self.n_frames = 5
        self.reduction = reduction
        self.freeze_backbone = freeze_backbone

        # ResNet18 feature backbone.
        backbone = resnet18(pretrained=pretrained_backbone)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # Channel reduction.
        self.conv1 = nn.Conv2d(512, 512 // reduction, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 512 // reduction)

        # Spatial depthwise transform.
        self.conv2 = nn.Conv2d(
            512 // reduction, 512 // reduction,
            kernel_size=3, padding=1,
            groups=512 // reduction, bias=False
        )

        # Global average pooling.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Temporal fusion module.
        if fusion_type == "mlp":
            self.attention_fusion = FusionMLP(embed_dim=512 // reduction)
        elif fusion_type == "transformer":
            # Create FusionTransformer from the provided config.
            if fusion_transformer_config is not None:
                self.attention_fusion = FusionTransformer(
                    embed_dim=512 // reduction,
                    num_heads=fusion_transformer_config.get("num_heads", 4),
                    num_layers=fusion_transformer_config.get("num_layers", 1),
                    dropout=fusion_transformer_config.get("dropout", 0.1),
                    droppath=fusion_transformer_config.get("droppath", 0.0)
                )
            else:
                self.attention_fusion = FusionTransformer(embed_dim=512 // reduction)
        else:
            raise ValueError("fusion_type must be 'mlp' or 'transformer'")

        # Learnable scaling for frame differences.
        self.scale = nn.Parameter(torch.ones(1))

        self.feature_dim = (512 // reduction) * 3 * 3

        # Cross-attention over memory and difference features.
        self.cross_attention = CrossAttention(
            dim=self.feature_dim,
            attn_drop=cross_attn_drop,
            proj_drop=cross_proj_drop)

        # Temporal attention over the frame sequence.
        self.temporal_attention = TemporalAttention(
            dim=self.feature_dim,
            num_heads=8,
            attn_drop=temp_attn_drop,
            proj_drop=temp_proj_drop
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.feature_extractor.eval()
        return self

    def forward(self, x):
        """
        Input: (B, 5, 3, H, W)
        Output: two (B, 128) features for memory and frame-difference dynamics.
        """
        B, T, C, H, W = x.size()
        assert T == self.n_frames, f"expected {self.n_frames} frames, got {T}"

        # Step 1: Feature extraction.
        x_flat = x.reshape(B * T, C, H, W)             # (B*5, 3, H, W)
        x_features = self.feature_extractor(x_flat)    # (B*5, 512, Hf, Wf)
        Hf, Wf = x_features.shape[-2:]

        # Step 2: Channel reduction.
        bottleneck = self.conv1(x_features)            # (B*5, C', Hf, Wf)
        bottleneck = self.norm1(bottleneck)

        # Step 3: Spatial transform.
        conv_bottleneck = self.conv2(bottleneck)       # (B*5, C', Hf, Wf)
        reshape_conv_bottleneck = conv_bottleneck.reshape(B, self.n_frames, -1, Hf, Wf)     # (B, 5, C', Hf, Wf)
        tPlusone_fea = reshape_conv_bottleneck[:, 1:]  # (B, 4, C', Hf, Wf)
        t_fea = reshape_conv_bottleneck[:, :-1]             # (B, 4, C', Hf, Wf)

        # Step 4: Scaled frame differencing.
        diff_fea = self.scale * (tPlusone_fea - t_fea)  # (B, 4, C', Hf, Wf)

        temporal_attn_out = self.temporal_attention(reshape_conv_bottleneck)  # (B, 5, C', Hf, Wf)

        B, T, C_, Hf, Wf = diff_fea.shape  # T=4
        D = C_ * Hf * Wf
        if D != self.feature_dim:
            raise ValueError(
                f"Expected flattened feature dim {self.feature_dim}, got {D}. "
                "Check the input image size and reduction ratio."
            )
        # reshape_conv_bottleneck: (B, 5, C', Hf, Wf) -> (B, 5, D)
        q = temporal_attn_out.reshape(B, self.n_frames, D)
        # diff_fea: (B, 4, C', Hf, Wf) -> (B, 4, D)
        kv = diff_fea.reshape(B, T, D)
        attn_out = self.cross_attention(q, kv)  # (B, 5, D)

        # Step 5: Pool spatial dimensions.
        # First reshape attn_out from (B, 5, D) back to (B, 5, C', Hf, Wf).
        attn_out_reshaped = attn_out.reshape(B, self.n_frames, C_, Hf, Wf)  # (B, 5, C', Hf, Wf)
        attn_out_flat = attn_out_reshaped.reshape(-1, C_, Hf, Wf)  # (B*5, C', Hf, Wf)
        memory_pooled_features = self.avg_pool(attn_out_flat).squeeze(-1).squeeze(-1)  # (B*5, C')
        # diff_fea: (B, 4, C', Hf, Wf) -> (B*4, C', Hf, Wf)
        diff_fea_flat = diff_fea.reshape(-1, C_, Hf, Wf)  # (B*4, C', Hf, Wf)
        diff_pooled_features = self.avg_pool(diff_fea_flat).squeeze(-1).squeeze(-1)  # (B*4, C')

        # Step 6: Temporal fusion.
        memory_pooled_features = memory_pooled_features.reshape(B, 5, -1)  # (B, 5, C')
        diff_pooled_features = diff_pooled_features.reshape(B, 4, -1)  # (B, 4, C')
        dynamic_memory_features = self.attention_fusion(memory_pooled_features)  # (B, C')
        dynamic_diff_features = self.attention_fusion(diff_pooled_features)  # (B, C')

        return dynamic_memory_features, dynamic_diff_features
