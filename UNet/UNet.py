import torch
import torch.nn as nn

from UNet.unet_blocks import ConvBlock, DownBlock, SelfAttentionBlock, UpBlock

class UNet(nn.Module):
    """Conditional UNet"""
    def __init__(self, in_channels: int, out_channels: int, encoder_decoder_layers: list, bottleneck_layers: list, UNet_embedding_dimensions: int, time_dimension: int, num_classes: int, device: str):
        """
        in_channels                 := Number of input channels
        out_channels                := Number of output channels
        encoder_decoder_layers      := Three values for the encoder/decoder
        bottleneck_layers           := Bottleneck dimension, at least one value
        UNet_embedding_dimensions   := Embedding dimensions for Up and Down modules
        time_dimension              := Time embedding dimension 
        num_classes                 := Number of classes in the dataset
        device                      := Which device the model is running on
        """
        super().__init__()
        self.time_dimension = time_dimension
        self.device = device

        # Setup class encoding
        self.label_embedding = nn.Embedding(num_classes, time_dimension)

        if encoder_decoder_layers:
            a, b, c = encoder_decoder_layers
        else:
            a = 64
            b = 128
            c = 256

        # Encoder TODO: Upgrade so we can have variable amount of down and self attention layers...
        self.initial_in     = ConvBlock(in_channels, a)
        down_1              = DownBlock(a, b, UNet_embedding_dimensions) # Halves the input size # 32
        self_attention_1    = SelfAttentionBlock(b)
        down_2              = DownBlock(b, c, UNet_embedding_dimensions)
        self_attention_2    = SelfAttentionBlock(c)
        down_3              = DownBlock(c, c, UNet_embedding_dimensions)
        self_attention_3    = SelfAttentionBlock(c)
        self.encoder        = nn.ModuleList([nn.ModuleList([down_1, self_attention_1]), nn.ModuleList([down_2, self_attention_2]), nn.ModuleList([down_3, self_attention_3])])

        # Bottle neck
        self.bottlenecks    = nn.ModuleList([ConvBlock(c, bottleneck_layers[0]),])
        for i in range(1, len(bottleneck_layers)-1):
            self.bottlenecks.append(ConvBlock(bottleneck_layers[i], bottleneck_layers[i]))

        self.bottlenecks.append(ConvBlock(bottleneck_layers[-1], c))

        # Decoder
        up_1                = UpBlock(2*c, b, UNet_embedding_dimensions) # Input to up is 2x since we have skip connection
        self_attention_4    = SelfAttentionBlock(b)
        up_2                = UpBlock(2*b, a, UNet_embedding_dimensions)
        self_attention_5    = SelfAttentionBlock(a)
        up_3                = UpBlock(2*a, a, UNet_embedding_dimensions)
        self_attention_6    = SelfAttentionBlock(a)
        self.decoder        = nn.ModuleList([nn.ModuleList([up_1, self_attention_4]), nn.ModuleList([up_2, self_attention_5]), nn.ModuleList([up_3, self_attention_6])])
        self.final_out      = nn.Conv2d(a, out_channels, kernel_size=1)


    def encode_positional_information(self, t: torch.Tensor):
        """Simple sinosodial encoding"""
        # https://arxiv.org/abs/1706.03762
        n = 10_000
        denom =  torch.pow(n, (torch.arange(0, self.time_dimension, 2, device=self.device).float() / self.time_dimension))
        sin_part = torch.sin(t.repeat(1, self.time_dimension // 2) / denom)
        cos_part = torch.cos(t.repeat(1, self.time_dimension // 2) / denom)
        encoding = torch.cat([sin_part, cos_part], dim=-1)
        return encoding
    

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        # Compute time encoding
        t = t.unsqueeze(-1) # Adds a dimension to our value
        t = self.encode_positional_information(t)

        # Append class encoding if it exsists
        if y is not None:
            t += self.label_embedding(y)
        
        # Forward pass
        encodings = [self.initial_in(x), ]
        for down, self_attention in self.encoder:
            _x = down(encodings[-1], t)
            _x = self_attention(_x)
            encodings.append(_x)

        for bottleneck in self.bottlenecks:
            encodings[-1] = bottleneck(encodings[-1])

        x = None
        for i, nodes in enumerate(self.decoder):
            up, self_attention = nodes
            if i == 0:
                x = up(encodings[-1], encodings[-2], t) #x4, x3
                x = self_attention(x)
            else:
                x = up(x, encodings[-2-i], t)
                x = self_attention(x)

        output = self.final_out(x)
        return output