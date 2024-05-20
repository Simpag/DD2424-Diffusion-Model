import torch
import torch.nn as nn

from UNet.unet_blocks import DoubleConv, Down, SelfAttention, Up

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
        self.initial_in     = DoubleConv(in_channels, a).to(self.device)
        down_1              = Down(a, b, UNet_embedding_dimensions).to(self.device) # Halves the input size # 32
        self_attention_1    = SelfAttention(b).to(self.device)   
        down_2              = Down(b, c, UNet_embedding_dimensions).to(self.device) 
        self_attention_2    = SelfAttention(c).to(self.device)
        down_3              = Down(c, c, UNet_embedding_dimensions).to(self.device)
        self_attention_3    = SelfAttention(c).to(self.device)
        self.encoder        = [[down_1, self_attention_1], [down_2, self_attention_2], [down_3, self_attention_3]]

        # Bottle neck
        self.bottlenecks    = [DoubleConv(c, bottleneck_layers[0]).to(self.device),]
        for i in range(1, len(bottleneck_layers)):
            self.bottlenecks.append(DoubleConv(bottleneck_layers[i], bottleneck_layers[i]).to(self.device))

        self.bottlenecks.append(DoubleConv(bottleneck_layers[-1], c).to(self.device))

        # Decoder
        up_1                = Up(2*c, b, UNet_embedding_dimensions).to(self.device) # Input to up is 2x since we have skip connection
        self_attention_4    = SelfAttention(b).to(self.device)
        up_2                = Up(2*b, a, UNet_embedding_dimensions).to(self.device)
        self_attention_5    = SelfAttention(a).to(self.device)
        up_3                = Up(2*a, a, UNet_embedding_dimensions).to(self.device)
        self_attention_6    = SelfAttention(a).to(self.device)
        self.decoder        = [[up_1, self_attention_4], [up_2, self_attention_5], [up_3, self_attention_6]]
        self.final_out      = nn.Conv2d(a, out_channels, kernel_size=1).to(self.device)


    def encode_positional_information(self, t: torch.Tensor):
        """Simple sinosodial encoding"""
        # https://arxiv.org/abs/1706.03762
        n = 10_000
        i_f = 1.0 / (
            torch.pow(n, (torch.arange(0, self.time_dimension, 2, device=self.device).float() / self.time_dimension)).to(self.device)
        )
        sin_part = torch.sin(t.repeat(1, self.time_dimension // 2) * i_f)
        cos_part = torch.cos(t.repeat(1, self.time_dimension // 2) * i_f)
        encoding = torch.cat([sin_part, cos_part], dim=-1)
        return encoding.to(self.device)
    

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