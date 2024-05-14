import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import Paint2CodeDataset
from utils import collate_fn, ids_to_tokens, generate_visualization_object, img_transformation
from modelArchitecture.modelCustomCNN import Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import pickle

# Configuration parameters
model_file_path = "./modelsCustomCNN/models/ED--epoch-85--loss-0.01651.pth"  #85 9851
data_path = Path("data", "all_data")
use_cuda = True
img_size = 224
split = "validation"
batch_size = 4
seed = 42

# Set random seeds for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
print(f"Using device: {device}")

# Ensure the model file exists
assert Path(model_file_path).exists(), "Model file path does not exist"

# Load the saved model
loaded_model = torch.load(model_file_path, map_location=device)
vocab = loaded_model["vocab"]

assert vocab is not None, "Vocabulary must be loaded."

# Initialize the models
embed_size = 64
hidden_size = 256
num_layers = 2

encoder = Encoder(embed_size).to(device)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)

# Load model weights
encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

encoder.eval()
decoder.eval()
print("Model loaded and set to evaluation mode.")

# Image transformations
transform_imgs = img_transformation(img_size)

# Data loader
data_loader = DataLoader(
    Paint2CodeDataset(data_path, split, vocab, transform=transform_imgs),
    batch_size=batch_size,
    collate_fn=lambda data: collate_fn(data, vocab=vocab),
    pin_memory=True if use_cuda else False,
    drop_last=False)

# Evaluate the model and calculate BLEU score
predictions = []
targets = []

for i, (image, caption) in enumerate(tqdm(data_loader.dataset)):
    image = image.to(device)
    caption = caption.to(device)
    features = encoder(image.unsqueeze(0))
    sample_ids = decoder.sample(features).cpu().data.numpy()
    
    predictions.append(ids_to_tokens(vocab, sample_ids))
    targets.append(ids_to_tokens(vocab, caption.cpu().numpy()))

bleu_score = corpus_bleu([[target] for target in targets], predictions, smoothing_function=SmoothingFunction().method4)
print(f"BLEU score: {bleu_score:.4f}")

exact_matches = sum([1 for pred, targ in zip(predictions, targets) if pred == targ])
exact_match_ratio = exact_matches / len(predictions)
print(f"Exact Match Ratio: {exact_match_ratio:.4f}")

total_correct_tokens = 0
total_tokens = 0

for pred, targ in zip(predictions, targets):
    # Предполагаем, что pred и targ уже являются списками токенов
    matched_tokens = sum(1 for p, t in zip(pred, targ) if p == t)
    total_correct_tokens += matched_tokens
    total_tokens += len(targ)

token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0
print(f"Token Accuracy: {token_accuracy:.4f}")