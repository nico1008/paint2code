import torch
import torch.nn as nn
import torch.nn.functional as F



class CustomCNN(nn.Module):
    def __init__(self, output_size):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dropout layers after each pooling
        self.dropout1 = nn.Dropout(0.15)  # Lower dropout after the first layer
        self.dropout2 = nn.Dropout(0.25)  # Moderate dropout after the second layer
        self.dropout3 = nn.Dropout(0.35)  # Higher dropout after the third layer
        
        # Activation
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=output_size)

    def forward(self, x):
        # Apply convolutions, activation function, pooling, and dropout
        x = self.dropout1(self.pool(self.activation(self.conv1(x))))
        x = self.dropout2(self.pool(self.activation(self.conv2(x))))
        x = self.dropout3(self.pool(self.activation(self.conv3(x))))
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with Activation and output
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.custom_cnn = CustomCNN(output_size=embedding_size)        
        self.BatchNorm = nn.BatchNorm1d(num_features=embedding_size, momentum=0.05)

    def forward(self, images):
        features = self.custom_cnn(images)
        features = self.BatchNorm(features)
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
