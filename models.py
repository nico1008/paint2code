import torch
import torch.nn as nn
import torchvision.models as models

#Paint2Code encoder
class Encoder(nn.Module):

    def __init__(self, embedding_size):

        super(Encoder, self).__init__()
        resnet = models.resnet18()
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Linear(in_features=resnet.fc.in_features, out_features=embedding_size)
        self.BatchNorm = nn.BatchNorm1d(num_features=embedding_size, momentum=0.05)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.BatchNorm(self.linear(features))
        return features


#Paint2Code decoder
class Decoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, features, captions, length):
        embeddings = self.embed(captions)

        features = features.unsqueeze(1)

        embeddings = torch.cat((features, embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=length, batch_first=True)
        hidden, _ = self.lstm(packed)
        output = self.linear(hidden[0])

        return output

    def sample(self, features, states=None, longest_sentence_length=100):

        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(longest_sentence_length):

            hidden, states = self.lstm(inputs, states)
            output = self.linear(hidden.squeeze(1))
            predicted = output.max(dim=1, keepdim=True)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.view(-1, 1, self.embed_size)

        sampled_ids = torch.cat(sampled_ids, 1)

        return sampled_ids.squeeze()
