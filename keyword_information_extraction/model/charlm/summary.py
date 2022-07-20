import torch

from pytorch_model_summary import summary
from keyword_information_extraction.model.charlm import CharacterLevelCNNHighwayBiLSTM as CharLM

if __name__ == "__main__":
    batch_size = 10
    n_entities = 80
    max_seq_length = 72

    n_classes = 5
    vocab_size = 68

    model_args = [
        ["char_embedding_dim", 15],
        ["char_conv_kernel_sizes", [1, 2, 3, 4, 5, 6]],
        ["char_conv_feature_maps", [25, 50, 75, 100, 125, 150]],  # [25 * kernel_size]
        ["num_highway_layers", 1],
        ["num_lstm_layers", 2],
        ["hidden_size", 300],
        ["dropout", 0.5],
        ["padding_idx", 0]
    ]

    args = dict(model_args)
    model = CharLM(n_classes=n_classes,
                   max_seq_length=max_seq_length,
                   char_vocab_size=vocab_size, **args)

    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, n_entities, max_seq_length)).float()
    model_summary = summary(model, inputs, max_depth=50, show_input=True)
    print(model_summary)
