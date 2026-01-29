"""
BIT (Binary Image Transformer) for Damage Change Detection
==========================================================

Implementation of the BIT model for damage change detection and segmentation.
Combines CNN backbones with transformer attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class ConvBNReLU(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize ConvBNReLU block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(self, backbone_name: str = "resnet50"):
        """
        Initialize ResNet backbone.
        
        Args:
            backbone_name: Name of the ResNet variant
        """
        super().__init__()
        if backbone_name not in models.__dict__:
            raise Exception(f"No model named {backbone_name} exists in torchvision.models.")
        
        backbone = models.__dict__[backbone_name](pretrained=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Set output channels based on backbone
        if backbone_name.startswith("resnet50") or backbone_name.startswith("resnet101") or backbone_name.startswith("resnet152"):
            self.out_channels = (256, 512, 1024, 2048)
        else:
            self.out_channels = (64, 128, 256, 512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning multi-scale features."""
        x = self.stem(x)
        c1 = x
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BITChangeDetector(nn.Module):
    """
    Binary Image Transformer for Damage Change Detection.
    
    Combines CNN backbones with transformer attention for accurate change detection.
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 depth: int = 6,
                 num_classes: int = 1,
                 dropout: float = 0.1):
        """
        Initialize BIT Change Detector.
        
        Args:
            backbone: CNN backbone name
            embed_dim: Transformer embedding dimension
            num_heads: Number of attention heads
            depth: Number of transformer blocks
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        # CNN backbones for pre and post images
        self.backbone_pre = ResNetBackbone(backbone)
        self.backbone_post = ResNetBackbone(backbone)
        
        # Feature fusion
        self.fusion_conv = ConvBNReLU(self.backbone_pre.out_channels[-1] * 2, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 32 * 32, embed_dim))  # Assuming 32x32 feature map
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize positional encoding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x1: Pre-event image tensor (B, 3, H, W)
            x2: Post-event image tensor (B, 3, H, W)
            
        Returns:
            Change detection mask (B, 1, H, W)
        """
        # Extract features from both images
        feats_pre = self.backbone_pre(x1)  # List of feature maps
        feats_post = self.backbone_post(x2)
        
        # Use highest level features for change detection
        feat_pre = feats_pre[-1]  # (B, C, H, W)
        feat_post = feats_post[-1]
        
        # Concatenate features
        feat_fused = torch.cat([feat_pre, feat_post], dim=1)  # (B, 2C, H, W)
        
        # Project to embedding dimension
        feat_embed = self.fusion_conv(feat_fused)  # (B, embed_dim, H, W)
        
        # Reshape for transformer
        B, C, H, W = feat_embed.shape
        feat_flat = feat_embed.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add positional encoding
        feat_flat = feat_flat + self.pos_embed[:, :H*W, :]
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            feat_flat = transformer_block(feat_flat)
        
        # Reshape back to spatial format
        feat_out = feat_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Generate segmentation mask
        output = self.seg_head(feat_out)
        
        # Upsample to original input size
        output = F.interpolate(output, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        return output
    
    def get_attention_maps(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            x1: Pre-event image tensor
            x2: Post-event image tensor
            
        Returns:
            Attention maps
        """
        # Extract features
        feats_pre = self.backbone_pre(x1)
        feats_post = self.backbone_post(x2)
        
        feat_pre = feats_pre[-1]
        feat_post = feats_post[-1]
        
        # Concatenate and embed
        feat_fused = torch.cat([feat_pre, feat_post], dim=1)
        feat_embed = self.fusion_conv(feat_fused)
        
        # Reshape
        B, C, H, W = feat_embed.shape
        feat_flat = feat_embed.flatten(2).transpose(1, 2)
        feat_flat = feat_flat + self.pos_embed[:, :H*W, :]
        
        # Collect attention maps
        attention_maps = []
        for transformer_block in self.transformer_blocks:
            attn = transformer_block.attn.norm1(feat_flat)
            attention_maps.append(attn)
            feat_flat = transformer_block(feat_flat)
        
        return attention_maps
