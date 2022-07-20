import torch

from typing import List, Optional


class CharacterLevelCNNHighwayBiLSTM(torch.nn.Module):

    def __init__(
            self,
            n_classes: int,
            max_seq_length: int,
            char_vocab_size: int,
            char_embedding_dim: int,
            char_conv_kernel_sizes: List[int],
            char_conv_feature_maps: List[int],
            num_highway_layers: int,
            num_lstm_layers: int,
            hidden_size: int,
            dropout: float,
            padding_idx: int
    ):
        r"""
        A simple neural language model that relies only on character-level inputs as described in:
        `Character-Aware Neural Language Models <https://arxiv.org/abs/1508.06615>`__.

        Args:
            n_classes (int): The number of classes.
            max_seq_length (int): The maximum sequence length.
            char_vocab_size (int): The size/dimension of the character embeddings.
            char_embedding_size (int): The size of each embedding vector.
            char_conv_kernel_sizes (int, list): The kernel/filter widths/sizes.
            char_conv_feature_maps (int, list): The number of filter/kernel matrices.
            use_batch_norm  (bool): A boolean that decides the use of BatchNorm1d.
            num_highway_layers (int): The number of highway networks.
            hidden_size (int): The number of features in the hidden state.
            dropout (float): A probability that equals to dropout.
                If non-zero, this will introduce a Dropout Layer after the specified outputs.
            padding_idx (int): If specified, the entries at `padding_idx` in the Embedding layer
                do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated
                during training, i.e. it remains as a fixed "pad".
        """
        super(CharacterLevelCNNHighwayBiLSTM, self).__init__()

        self.hidden_size: int = hidden_size

        self.char_embeddings: torch.nn.Module = torch.nn.Embedding(num_embeddings=char_vocab_size + 1,
                                                                   embedding_dim=char_embedding_dim,
                                                                   # As we have the token padding value
                                                                   # we also need to set it in the character embedding.
                                                                   # And from the definition of 'padding_idx':
                                                                   # padding_idx pads the output
                                                                   # with the embedding vector at padding_idx
                                                                   # (initialized to zero) whenever
                                                                   # it encounters the index.
                                                                   padding_idx=padding_idx)

        self.pool_layers: torch.nn.Module = torch.nn.ModuleList(
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

        # Excluding learnable parameters.
        self.batch_norm1d: torch.nn.Module = torch.nn.BatchNorm1d(highway_input_dim, affine=False)

        self.highway_layers: torch.nn.Module = torch.nn.Sequential(
            *[HighwayNetwork(input_size=highway_input_dim) for _ in range(num_highway_layers)]
        )

        self.bilstm: torch.nn.Module = torch.nn.LSTM(
            input_size=highway_input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bias=True,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size * 2, n_classes)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for name, layer in self.named_children():
            if name not in ("highway_layers", "bilstm_layer"):
                for module in layer.modules():
                    for params in module.parameters():
                        torch.nn.init.uniform_(params.data, a=-0.05, b=0.05)
            elif name == "bilstm_layer":
                # Initialize the BLSTM weights as described in:
                # `On orthogonality and learning recurrent networks with long term dependencies`
                # <https://arxiv.org/abs/1702.00071>`_
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

        # Input's shape: [B, N, L]
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

        self.affine_transformation = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size, bias=True),
            torch.nn.ReLU()
        ).apply(HighwayNetwork.init_affine_transformation)

        self.transform_gate = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size, bias=True),
            torch.nn.ReLU()
        ).apply(HighwayNetwork.init_transform_gate)

    def forward(self, y):
        t = self.transform_gate(y)
        g = self.affine_transformation(y)
        # z = t * g(Why + bh) + (1 - t) * y
        z = t * g + (1.0 - t) * y
        return z

    @staticmethod
    def init_transform_gate(layer):
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.uniform_(layer.weight, a=-0.05, b=0.05)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=-2.0)

    @staticmethod
    def init_affine_transformation(layer):
        if isinstance(layer, torch.nn.Linear):
            for param in layer.parameters():
                torch.nn.init.uniform_(param.data, a=-0.05, b=0.05)
