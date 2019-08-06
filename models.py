import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)

        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        
        self.hidden2notes = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        print(x.shape)
        e_embedding = self.embedding(x)
        _, (ht, ct) = self.encoder(e_embedding)

        d_embedding = self.decoder_embedding(x)
        print(d_embedding.shape)
        d_embedding = torch.cat(d_embedding, dim=2)
        print(d_embedding.shape)
        decode = self.decoder(d_embedding, (ht, ct))
        output = self.hidden2notes(decode)
        return output
