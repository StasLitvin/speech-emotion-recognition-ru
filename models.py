import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, TRAIN


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class CNN2D(nn.Module):


    def __init__(self, num_classes=NUM_CLASSES, dropout=TRAIN.dropout):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class Attention(nn.Module):


    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        scores = self.context(torch.tanh(self.proj(h)))
        alpha = F.softmax(scores, dim=1)
        return (alpha * h).sum(dim=1)


class CNN_BiLSTM_Attention(nn.Module):


    def __init__(self, num_classes=NUM_CLASSES, lstm_hidden=128,
                 dropout=TRAIN.dropout):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 1)),
            ConvBlock(32, 64, pool=(2, 1)),
            ConvBlock(64, 128, pool=(2, 2)),
        )

        self.lstm = nn.LSTM(input_size=128 * (128 // 8),
                            hidden_size=lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.attn = Attention(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h, _ = self.lstm(x)
        ctx = self.attn(h)
        return self.classifier(ctx)


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "cnn2d":
        return CNN2D()
    if name in ("cnn_bilstm", "cnn-bilstm", "bilstm"):
        return CNN_BiLSTM_Attention()
    raise ValueError(f"Неизвестная архитектура: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
