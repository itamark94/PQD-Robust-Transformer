import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Callable
from torch.nn.modules.container import ModuleList


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer1D(nn.Module):
    """
    Pre-LN Transformer encoder model for PQD classification.

    Args:
        d_model: number of expected features in the inputs.
        nhead: number of heads used in the multi-head self-attention mechanism.
        dim_feedforward: dimension of the feed-forward neural network model.
        dropout: dropout rate on the output of the feed-forward.
        dropout_attn: dropout rate used in the multi-head self-attention mechanism.
        n_blocks: number of encoder layers.
        n_classes: number of classes (types of PQDs including normal).
        return_attn_maps: bool which determines whether the forward function also returns the attention rollout.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, dropout_attn: float,
                 n_blocks: int, n_classes: int, return_attn_maps: bool):
        super(Transformer1D, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, dropout_attn,
                                                      norm_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, n_blocks, return_attn_maps=return_attn_maps)
        self.out = nn.Linear(self.d_model, self.n_classes)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [batch_size, 1, length] --> [length, batch_size, 1]
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)[0]
        x = x.mean(dim=0)  # [length, batch_size, d_model] --> [batch_size, d_model]
        x = self.out(x)
        return x

    def forward_and_attention_rollout(self, x):
        x = x.permute(2, 0, 1)  # [batch_size, 1, length] --> [length, batch_size, 1]
        batch_size, length, _ = x.shape
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x, attention_maps = self.transformer_encoder(x)
        rollout = attention_rollout(attention_maps)
        x = x.mean(dim=0)  # [length, batch_size, d_model] --> [batch_size, d_model]
        x = self.out(x)
        return x, rollout


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_attn_maps=False):
        """
        Custom TransformerEncoder module based on the open source code in pytorch. The customization allows the forward
        function to also return the attention maps.

        Args:
            encoder_layer: an instance of the TransformerEncoderLayer() class (required).
            num_layers: number of encoder-layers (required).
            norm: the layer normalization component (optional).
            return_attn_maps: determines whether the forward function returns also a list of the attention maps
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_attn_maps = return_attn_maps

    def forward(self, src: Tensor) -> [Tensor, [Tensor]]:
        output = src
        attention_maps = []

        if self.return_attn_maps:
            for mod in self.layers:
                output, attn_map = mod(output)
                attention_maps.append(attn_map)
        else:
            for mod in self.layers:
                output = mod(output)[0]

        if self.norm is not None:
            output = self.norm(output)

        return output, attention_maps


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom TransformerEncoderLayer module based on the open source code in pytorch, implements both Pre-LN and Post-LN
    Transformer encoder layer.

    Args:
        d_model: the number of expected features in the inputs.
        nhead: the number of heads used in the multi-head self-attention mechanism.
        dim_feedforward: the dimension of the feed-forward neural network model.
        dropout: the dropout value on the output of the feed-forward.
        dropout_attn: the dropout value used in the multi-head self-attention mechanism.
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, dropout_attn: float,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn, bias=bias, batch_first=batch_first,
                                               **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(self, src: Tensor) -> [Tensor, Tensor]:
        if self.norm_first:  # Pre-LN
            x, attn_map = self._sa_block(self.norm1(src))
            x = x + src
            x = x + self._ff_block(self.norm2(x))
        else:  # Post-LN
            x, attn_map = self._sa_block(src)
            x = self.norm1(x + src)
            x = self.norm2(x + self._ff_block(x))
        return x, attn_map

    def _sa_block(self, x: Tensor) -> [Tensor, Tensor]:
        x, a = self.self_attn(x, x, x)
        x = self.dropout1(x)
        return x, a

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def attention_rollout(attn_maps: [Tensor]) -> Tensor:
    """
    Computes attention rollout based on the paper "Quantifying attention flow in transformers".

    Args:
        attn_maps: list of attention maps
    """
    batch_size, length, _ = attn_maps[0].shape
    eye = torch.eye(length).expand(batch_size, length, length).to(attn_maps[0].device)
    rollout = eye
    for a in attn_maps:
        a = (a + 1.0 * eye) / 2
        rollout = torch.bmm(a, rollout)
    return rollout


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return lambda x: F.relu(x, inplace=False)
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


########################################################################################################################


class DeepCNN(nn.Module):
    def __init__(self, n_classes=10):
        """
        DeepCNN PQD classifier which was used in the paper "Open source dataset generator for power quality disturbances
        with deep-learning reference classifiers", and also used in the paper "Neural Architecture Search (NAS) for
        Designing Optimal Power Quality Disturbance Classifiers". The model was first introduced in the paper "A novel
        deep learning method for the classification of power quality disturbances using deep convolutional neural
        network". In this work ("Improving robustness of Transformers for power quality disturbance classification via
        optimized relevance maps"), the model is used to compare classification performance under adversarial attacks.
        """
        super(DeepCNN, self).__init__()

        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=3 // 2)
        self.block1_conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=3 // 2)
        self.block1_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2)
        self.block1_bn = nn.BatchNorm1d(16)

        self.block2_conv1 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3 // 2)
        self.block2_conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=3 // 2)
        self.block2_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2)
        self.block2_bn = nn.BatchNorm1d(32)

        self.block3_conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.block3_conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.block3_pool = nn.AdaptiveMaxPool1d(1)
        self.block3_bn = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=n_classes)


    def forward(self, x):
        out = F.relu(self.block1_conv1(x), inplace=True)
        out = F.relu(self.block1_conv2(out), inplace=True)
        out = self.block1_pool(out)
        out = self.block1_bn(out)

        out = F.relu(self.block2_conv1(out), inplace=True)
        out = F.relu(self.block2_conv2(out), inplace=True)
        out = self.block2_pool(out)
        out = self.block2_bn(out)

        out = F.relu(self.block3_conv1(out), inplace=True)
        out = F.relu(self.block3_conv2(out), inplace=True)
        out = self.block3_pool(out)
        out = self.block3_bn(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = self.bn(out)
        out = self.fc2(out)

        return out


class CNN(nn.Module):
    """
    CNN PQD classifier, a smaller version of the DeepCNN and in this work ("Improving robustness of Transformers for
    power quality disturbance classification via optimized relevance maps") serves as the surrogate model for black-box
    adversarial attacks.
    """
    def __init__(self, n_classes=10):
        super(CNN, self).__init__()

        self.block1_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=3 // 2)
        self.block1_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2)
        self.block1_bn = nn.BatchNorm1d(16)

        self.block2_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3 // 2)
        self.block2_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=3 // 2)
        self.block2_bn = nn.BatchNorm1d(32)

        self.block3_conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2)
        self.block3_pool = nn.AdaptiveMaxPool1d(1)
        self.block3_bn = nn.BatchNorm1d(64)

        self.fc = nn.Linear(in_features=64, out_features=n_classes)


    def forward(self, x):
        out = F.relu(self.block1_conv(x), inplace=True)
        out = self.block1_pool(out)
        out = self.block1_bn(out)

        out = F.relu(self.block2_conv(out), inplace=True)
        out = self.block2_pool(out)
        out = self.block2_bn(out)

        out = F.relu(self.block3_conv(out), inplace=True)
        out = self.block3_pool(out)
        out = self.block3_bn(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
