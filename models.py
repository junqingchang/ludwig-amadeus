import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               num_layers, bidirectional=True, batch_first=True)

        self.hidden2notes = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, x):
        e_embedding = self.embedding(x)
        lstm_out, (hn, cn) = self.encoder(e_embedding)
        output = self.hidden2notes(hn.reshape([x.shape[0], 1, -1]))
        tag_scores = F.log_softmax(output, dim=2)
        tag_scores = tag_scores.transpose(1, 2)
        return tag_scores
