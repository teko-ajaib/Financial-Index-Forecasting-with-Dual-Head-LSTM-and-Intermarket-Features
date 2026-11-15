import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesSequenceDataset(Dataset):
    def __init__(self, data_scaled, raw_returns=None, seq_len=10, target_idx=0):
        self.seq_len = seq_len
        self.target_idx = target_idx

        n_rows, _ = data_scaled.shape
        X_list, y_reg_list, y_dir_list = [], [], []

        for i in range(seq_len, n_rows):
            window = data_scaled[i - seq_len: i, :]
            target_scaled_ret = data_scaled[i, target_idx]

            if raw_returns is not None:
                is_up = 1.0 if raw_returns[i, target_idx] > 0 else 0.0
            else:
                is_up = 1.0 if target_scaled_ret > 0 else 0.0

            X_list.append(window)
            y_reg_list.append([target_scaled_ret])
            y_dir_list.append([is_up])

        self.X = torch.FloatTensor(np.array(X_list))
        self.y_reg = torch.FloatTensor(np.array(y_reg_list))
        self.y_dir = torch.FloatTensor(np.array(y_dir_list))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_reg[idx], self.y_dir[idx]


class DualHeadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(DualHeadLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.dir_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        reg_out = self.reg_head(last_step)
        dir_out = self.dir_head(last_step)
        return reg_out, dir_out


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.model = DualHeadLSTM(input_size, hidden_size, num_layers, dropout)

    def forward(self, x):
        reg, _ = self.model(x)
        return reg
