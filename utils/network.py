import torch.nn as nn


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def get_norm_layer(norm_layer="", dim=-1):
    norm_layers = {
        "": nn.Identity(),
        "batch_norm": nn.BatchNorm1d(dim),
        "layer_norm": nn.LayerNorm(dim)
    }

    return norm_layers[norm_layer]


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2, norm_layer="batch_norm"):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_norm_layer(norm_layer, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, output_dim),
            get_norm_layer(norm_layer, output_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        out = self.block(x) + x
        return out


class BasicBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2, norm_layer="batch_norm"):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_norm_layer(norm_layer, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, output_dim),
            get_norm_layer(norm_layer, output_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, p_dropout=0.2, norm_layer="batch_norm"):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_norm_layer(norm_layer, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            BasicBlock(hidden_dim, hidden_dim, hidden_dim, p_dropout=p_dropout, norm_layer=norm_layer),
            BasicBlock(hidden_dim, hidden_dim, hidden_dim, p_dropout=p_dropout, norm_layer=norm_layer),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class SMPLHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2, norm_layer="batch_norm"):
        super().__init__()

        self.block = nn.Sequential(
            BasicBlock(input_dim, hidden_dim, hidden_dim, p_dropout=p_dropout, norm_layer=norm_layer),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class DomainCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, norm_layer="batch_norm"):
        super().__init__()

        output_dim = 1

        self.block = nn.Sequential(
            BasicBlock2(input_dim, hidden_dim, hidden_dim, norm_layer=norm_layer),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, step=0):
        out = self.block(x)
        return out
