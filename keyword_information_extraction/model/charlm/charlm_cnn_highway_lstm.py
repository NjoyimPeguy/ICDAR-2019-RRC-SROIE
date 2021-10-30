import torch

from typing import List


class CharacterLevelCNNHighwayBiLSTM(torch.nn.Module):
    
    def __init__(
            self,
            n_classes: int,
            max_seq_length: int,
            char_vocab_size: int,
            char_embedding_dim: int,
            char_conv_kernel_sizes: List[int],
            char_conv_feature_maps: List[int],
            use_batch_norm: bool = False,
            num_highway_layers: int = 1,
            num_lstm_layers: int = 1,
            hidden_size: int = 256,
            dropout: float = 0.0,
            padding_idx: int = 0
    ):
        """
        A simple neural language model that relies only on character-level inputs which is based on this paper:
        https://arxiv.org/pdf/1508.06615.pdf
        
        Args:
            n_classes: The number of classes.
            max_seq_length: The maximum sequence length.
            char_vocab_size: The size/dimension of the character embeddings.
            char_embedding_size: The size of each embedding vector.
            char_conv_kernel_sizes: The kernel/filter widths/sizes.
            char_conv_feature_maps: The number of filter/kernel matrices.
            use_batch_norm: A boolean that decides the use of BatchNorm1d.
            num_highway_layers: The number of highway networks.
            hidden_size: The number of features in the hidden state.
            dropout: A probability that equals to dropout.
                If non-zero, this will introduce a Dropout Layer after the specified outputs.
        """
        super(CharacterLevelCNNHighwayBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        
        self.char_embeddings = torch.nn.Embedding(num_embeddings=char_vocab_size + 1,
                                                  embedding_dim=char_embedding_dim,
                                                  # As we have the token padding value we also need to set it in the
                                                  # character embedding. And from the definition of 'padding_idx':
                                                  # padding_idx pads the output with the embedding vector at
                                                  # padding_idx (initialized to zeros) whenever it encounters the
                                                  # index.
                                                  padding_idx=padding_idx)
        
        self.pool_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_channels=char_embedding_dim,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        bias=True
                    ),
                    torch.nn.Tanh(),
                    torch.nn.MaxPool1d(kernel_size=max_seq_length - kernel_size + 1, stride=1, padding=0)
                )
                for kernel_size, out_channels in zip(char_conv_kernel_sizes, char_conv_feature_maps)
            ]
        )
        
        highway_input_dim = sum(char_conv_feature_maps)
        
        self.batch_norm1d = torch.nn.BatchNorm1d(highway_input_dim, affine=False) if use_batch_norm else None
        
        self.highway_layers = torch.nn.Sequential(
            *[HighwayNetwork(input_size=highway_input_dim) for _ in range(num_highway_layers)]
        )
        
        self.bilstm = torch.nn.LSTM(
            input_size=highway_input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bias=True,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        
        # The classifier layer.
        self.output_layer = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size * 2, n_classes)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, layer in self.named_children():
            if name not in ("highway_layers", "bilstm"):
                for params in layer.parameters():
                    torch.nn.init.uniform_(params.data, a=-0.05, b=0.05)
            elif name == "bilstm":
                # init the Bi-LSTM layer based on this paper https://arxiv.org/pdf/1702.00071.pdf
                # if len(param.shape) >= 2, then it is 'the weight_ih_l{}{}' or 'weight_hh_l{}{}'
                # Otherwise it is the 'bias_ih_l{}{}', 'bias_hh_l{}{}'
                for param in layer.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.zeros_(param.data)
                        torch.nn.init.ones_(param.data[self.hidden_size:self.hidden_size * 2])
    
    def forward(self, char_inputs: torch.tensor):
        """
        Apply the forward propagation.
        
        Args:
            char_inputs: An input tensor whose shape is [B, N, L] where
                         B: the batch size,
                         N: the number of sequences,
                         L: the length of a given sequence which the maximum sequence length.
                         
        Returns:
            A tensor of shape [B, N, #classes]
        """
        
        batch_size, num_sequences, max_seq_length = char_inputs.shape
        
        # Input's shape from [B, N, L] -> [BxN, L]
        char_inputs = char_inputs.contiguous().view(-1, max_seq_length)
        
        # Embedded feature shape: [BxN, L, char_embedding_dim]
        char_embedded_features = self.char_embeddings(char_inputs.long())
        
        # Turning the char inputs' shape into a shape acceptable by the 1D-CNN layers.
        # [BxN, L, char_embedding_dim] -> [BxN, char_embedding_dim, L]
        char_embedded_features = char_embedded_features.transpose(1, 2).float()
        
        pool_layers = []
        for pool_layer in self.pool_layers:
            feature_map = pool_layer(char_embedded_features).squeeze()
            pool_layers.append(feature_map)
        
        # CNN features shape: [BxN, total_num_filters]
        # where total_num_filters is the sum over the CNN 1D output channels
        cnn_features = torch.cat(pool_layers, dim=1)
        
        if self.use_batch_norm:
            # Shape: [BxN, total_num_filters]
            cnn_features = self.batch_norm1d(cnn_features)
        
        # Highway features shape: [BxN, total_num_filters]
        highway_features = self.highway_layers(cnn_features)
        
        # Turning the highway features' shape into a shape acceptable by the LSTM layers.
        # [BxN, total_num_filters] -> [B, N, total_num_filters]
        highway_features = highway_features.contiguous().view(batch_size, num_sequences, -1)
        
        # LSTM features shape: [B, N, hidden_size]
        lstm_features, _ = self.bilstm(highway_features)
        
        # Output features shape: [B, N, #classes]
        output_features = self.output_layer(lstm_features)
        
        return output_features


class HighwayNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(HighwayNetwork, self).__init__()
        
        self.transform_gate = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size, bias=True),
            torch.nn.ReLU()
        ).apply(init_transform_gate)
        
        self.affine_transformation = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size, bias=True),
            torch.nn.ReLU()
        ).apply(init_affine_transformation)
    
    def forward(self, y):
        t = self.transform_gate(y)
        g = self.affine_transformation(y)
        carry_gate = 1 - t
        # z = t * g(Why + bh) + (1 - t) * y
        z = (t * g) + (carry_gate * y)
        return z


def init_transform_gate(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.uniform_(layer.weight, a=-0.05, b=0.05)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=-2.0)


def init_affine_transformation(layer):
    if isinstance(layer, torch.nn.Linear):
        for param in layer.parameters():
            torch.nn.init.uniform_(param.data, a=-0.05, b=0.05)


if __name__ == '__main__':
    import torch.nn.functional as F
    from pytorch_model_summary import summary
    
    batch_size = 8
    n_sentences = 35
    max_seq_length = 72
    
    n_classes = 5
    vocab_size = 68
    
    model_args = [
        ["char_embedding_dim", 15],
        ["char_conv_kernel_sizes", [1, 2, 3, 4, 5, 6]],
        ["char_conv_feature_maps", [25, 50, 75, 100, 125, 150]],  # [25 * kernel_size]
        ["use_batch_norm", True],
        ["num_highway_layers", 1],
        ["num_lstm_layers", 2],
        ["hidden_size", 300],
        ["dropout", 0.5]
    ]
    
    args = dict(model_args)
    model = CharacterLevelCNNHighwayBiLSTM(n_classes=n_classes,
                                           max_seq_length=max_seq_length,
                                           char_vocab_size=vocab_size, **args)
    
    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, n_sentences, max_seq_length)).float()
    model_summary = summary(model, inputs)
    print(model_summary)
