import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        """
        Initializes the SelfAttentionBlock with multi-head attention and normalization layers.

        Parameters:
            channels := Number of input channels
        """
        super().__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.sequence = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )


    def forward(self, x: torch.Tensor):
        """
        Forward pass for the SelfAttentionBlock.

        Parameters:
            x := Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor after applying self-attention and transformation layers.
        """
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.sequence(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate_channels=None, skip_connection=False):
        """
        Initializes the ConvBlock with two convolutional layers and optional skip connection.

        Parameters:
            in_channels          := Number of input channels
            out_channels         := Number of output channels
            intermediate_channels:= Number of intermediate channels (optional)
            skip_connection      := Whether to use skip connection (default: False)
        """
        super().__init__()
        self.skip_connection = skip_connection

        if not intermediate_channels:
            intermediate_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, intermediate_channels),
            nn.GELU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )


    def forward(self, x):
        """
        Forward pass for the ConvBlock.

        Parameters:
            x := Input tensor

        Returns:
            Tensor after applying convolutional layers and optional skip connection.
        """
        if self.skip_connection:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dimension):
        """
        Initializes the DownBlock with max-pooling and convolutional layers, along with embedding layer.

        Parameters:
            in_channels        := Number of input channels
            out_channels       := Number of output channels
            embedding_dimension:= Dimension of the embedding vector
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, skip_connection=True),
            ConvBlock(in_channels, out_channels),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embedding_dimension,
                out_channels
            ),
        )


    def forward(self, x, t):
        """
        Forward pass for the DownBlock.

        Parameters:
            x := Input tensor
            t := Embedding tensor

        Returns:
            Tensor after applying down-sampling, convolutional layers, and embedding addition.
        """
        x = self.maxpool_conv(x)
        emb = self.embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dimension):
        """
        Initializes the UpBlock with up-sampling and convolutional layers, along with embedding layer.

        Parameters:
            in_channels        := Number of input channels
            out_channels       := Number of output channels
            embedding_dimension:= Dimension of the embedding vector
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, skip_connection=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                embedding_dimension,
                out_channels
            ),
        )


    def forward(self, x, skip_x, t):
        """
        Forward pass for the UpBlock.

        Parameters:
            x      := Input tensor
            skip_x := Skip connection tensor
            t      := Embedding tensor

        Returns:
            Tensor after applying up-sampling, concatenation with skip connection, convolutional layers, and embedding addition.
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # add the skip connection
        x = self.conv(x)
        emb = self.embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb