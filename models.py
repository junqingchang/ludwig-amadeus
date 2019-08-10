import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)

        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        
        self.hidden2notes = nn.Linear(hidden_dim*2, vocab_size)

    def forward(self, encode, decode):
        e_embedding = self.embedding(encode)
        _, (hn, cn) = self.encoder(e_embedding)

        d_embedding = self.decoder_embedding(decode)
        decode, _ = self.decoder(d_embedding, (hn[:,-1,:].unsqueeze(1).contiguous(), cn[:,-1,:].unsqueeze(1).contiguous()))
        output = self.hidden2notes(decode)
        tag_scores = F.log_softmax(output, dim=2)
        tag_scores = tag_scores.transpose(1,2)
        return tag_scores
