import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, input_size=88, d_model=128, d_ff=2048, nhead=8, num_layers=6, output_size=88, max_seq_length=100,
                 dropout=0.1, padding_value=-99, device='cpu'):
        super(Transformer, self).__init__()

        self.device = device
        self.padding_value = padding_value

        self.linear_embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=d_ff, dropout=dropout, 
                                                    activation='relu', layer_norm_eps=1e-05, 
                                                    batch_first=True, norm_first=False, bias=True, 
                                                    device=device, dtype=None)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.fc_activation = nn.Sigmoid()

    def forward(self, src):
        # input embedding
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1), device=self.device, dtype=torch.float32)
        src = self.dropout(self.positional_encoding(self.linear_embedding(src)))

        # transformer
        output = self.transformer_encoder(src=src, mask=src_mask, is_causal=True)

        # linear decoder
        output = self.fc_activation(self.fc(output))

        return output

    def criterion(self, pred, tgt):  # loss function
        output_size = tgt.size(-1)
        mask = torch.where(tgt == self.padding_value, 0, 1)  #get mask for padding values
        pred, tgt = pred * mask, tgt * mask             
        loss = nn.BCELoss(reduction='none')(pred, tgt)  #compute BCELoss
        loss = loss * mask                              #masking the padding values in the output loss
        loss = loss.sum() / mask.sum()                  #mean of the loss
        loss *= output_size                             #scaling the loss to retrieve average loss per timestep

        return loss


class LSTM(nn.Module):
    def __init__(self, input_size=88, n_lstm_layers=3, hidden_dim=88, output_size=88,dropout = 0.1,padding_value = -99, device = 'cpu'):
        super(LSTM, self).__init__()

        self.device = device
        self.padding_value = padding_value

        self.n_lstm_layers = n_lstm_layers
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_dim, n_lstm_layers, dropout = dropout, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, output_size)
        self.output_fc_activation = nn.Sigmoid()

    def _initialize_weights(self, mean=0.0, std=0.1):
        nn.init.normal_(self.output_fc.weight, mean=mean, std=std)
        nn.init.normal_(self.output_fc.bias, mean=mean, std=std)
        for name, param in self.named_parameters():
            if 'lstm.weight' in name:
                nn.init.normal_(param, mean=mean, std=std)
            if 'lstm.bias' in name:
                nn.init.normal_(param, mean=mean, std=std)

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim, device = self.device)
        c_0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim, device = self.device)
        return (h_0, c_0)

    def forward(self, X, hidden):
        batch_size, seq_len, _ = X.size()

        X_lengths = torch.where(X[:,:,0] != self.padding_value, 1, 0).sum(1).to(torch.device('cpu'))
        X_lengths_clamped = X_lengths.clamp(min=1, max=X_lengths.max())

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths_clamped, batch_first=True,enforce_sorted=False)
        X, hidden = self.lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        X = self.output_fc(X)
        X = self.output_fc_activation(X)
        X = X.view(batch_size, seq_len, self.output_size)

        return X, (hidden)

    def criterion(self, pred, tgt): #loss function
        output_size = tgt.size(-1)
        mask = torch.where(tgt == self.padding_value, 0, 1)  #get mask for padding values
        pred, tgt = pred * mask, tgt * mask             
        loss = nn.BCELoss(reduction='none')(pred, tgt)  #compute BCELoss
        loss = loss * mask                              #masking the padding values in the output loss
        loss = loss.sum() / mask.sum()                  #mean of the loss
        loss *= output_size                             #scaling the loss to retrieve average loss per timestep

        return loss
