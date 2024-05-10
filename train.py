import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm  

from vocab import Vocab
from dataset import Paint2CodeDataset
from utils import collate_fn, save_model, img_transformation
from modelResnet18 import Encoder, Decoder

# Set parameters
data_path = Path("data", "all_data")
vocab_file_path = None or Path(data_path.parent, "vocab.txt")
use_cuda = True  
img_crop_size = 224
split = "train"
save_after_epochs = 5
models_dir = Path("models")
batch_size = 4
epochs = 17
lr = 0.001
seed = 42

# Seed setup for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

vocab = Vocab(vocab_file_path)
assert len(vocab) > 0, "Vocabulary must be non-empty"

# Setup GPU for training
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
print(f"Using device: {device}")

transform_imgs = img_transformation(img_crop_size)

train_loader = DataLoader(
    Paint2CodeDataset(data_path, split, vocab, transform=transform_imgs),
    batch_size=batch_size,
    collate_fn=lambda data: collate_fn(data, vocab=vocab),
    pin_memory=True if use_cuda else False,
    drop_last=True
)
print("Data loader for train created successfully.")

test_loader = DataLoader(
    Paint2CodeDataset(data_path,"test", vocab, transform=transform_imgs),
    batch_size=batch_size,
    collate_fn=lambda data: collate_fn(data, vocab=vocab),
    pin_memory=True if use_cuda else False,
    drop_last=True
)
print("Data loader for test created successfully.")

# Model parameters
embed_size = 256
hidden_size = 512
num_layers = 1

# Initialize the Encoder and Decoder
encoder = Encoder(embed_size).to(device)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)

print("Models are initialized and moved to the designated device.")

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.BatchNorm.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

print("Loss function and optimizer are set.")

# Initialize lists to store the average losses per epoch
train_losses = []
test_losses = []

print("Starting training...")
for epoch in tqdm(range(epochs)):
    # Training phase
    encoder.train()  # Set the encoder to training mode
    decoder.train()  # Set the decoder to training mode
    total_train_loss = 0
    num_batches = 0

    for images, captions, lengths in train_loader:
        images = images.to(device)
        captions = captions.to(device)
        targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Zero the gradients
        encoder.zero_grad()
        decoder.zero_grad()

        # Forward pass
        features = encoder(images)
        output = decoder(features, captions, lengths)

        # Calculate loss
        loss = criterion(output, targets)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        num_batches += 1

    avg_train_loss = total_train_loss / num_batches
    train_losses.append(avg_train_loss)  # Append the average training loss

    # Testing phase
    encoder.eval()  # Set the encoder to evaluation mode
    decoder.eval()  # Set the decoder to evaluation mode
    total_test_loss = 0
    num_test_batches = 0

    with torch.no_grad():  # No need to track gradients during testing
        for images, captions, lengths in test_loader:
            images = images.to(device)
            captions = captions.to(device)
            targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward pass
            features = encoder(images)
            output = decoder(features, captions, lengths)

            # Calculate loss
            loss = criterion(output, targets)
            total_test_loss += loss.item()
            num_test_batches += 1

    avg_test_loss = total_test_loss / num_test_batches
    test_losses.append(avg_test_loss)  # Append the average testing loss

    # Print integrated loss statistics in one line
    print(f'Epoch {epoch}: Training Loss {avg_train_loss:.3f}, Test Loss {avg_test_loss:.3f}')

    #Save model checkpoint
    if epoch != 0 and epoch % save_after_epochs == 0:
        save_model(models_dir, encoder, decoder, optimizer, epoch, avg_train_loss, batch_size, vocab)
        print("Saved model checkpoint at epoch:", epoch)

print("Training completed!")

#Save the final model
save_model(models_dir, encoder, decoder, optimizer, epochs - 1, loss.item(), batch_size, vocab)
print("Final model saved.")