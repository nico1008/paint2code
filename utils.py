import torch
from pathlib import Path
import pickle
from torchvision import transforms

def collate_fn(data, vocab):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: List of tuples (image: torch.Tensor, caption: torch.Tensor).
        vocab: Vocabulary object with token methods.

    Returns:
        Tuple of (images, targets, lengths) where:
        images is a tensor of shape (batch_size, 3, 256, 256),
        targets is a tensor of shape (batch_size, max_length),
        lengths is a list of valid caption lengths.
    """
    if not data or not vocab:
        raise ValueError("Data and vocab must be provided and not empty.")

    # Sort data by descending caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    target_lengths = max(lengths)
    padding_token_id = vocab.get_id_by_token(vocab.get_padding_token())
    targets = torch.full((len(captions), target_lengths), padding_token_id, dtype=torch.long)
    
    for i, cap in enumerate(captions):
        targets[i, :lengths[i]] = cap

    return images, targets, lengths


def img_transformation(img_crop_size):
    return transforms.Compose([transforms.Resize((img_crop_size, img_crop_size)),
                               transforms.ToTensor()])


def save_model(models_folder_path, encoder, decoder, optimizer, epoch, loss, batch_size, vocab):
    MODELS_FOLDER = Path(models_folder_path)

    # Create the models folder if it's not already there
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    rounded_loss = f'{loss:.5f}'
    model_name = f"ED--epoch-{epoch}--loss-{rounded_loss}.pth"
    MODEL_PATH = MODELS_FOLDER / model_name

    torch.save({'epoch': epoch,
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'vocab': vocab
                }, MODEL_PATH)

def ids_to_tokens(vocab, ids):
    tokens = []

    for id in ids:
        token = vocab.get_token_by_id(id)

        if token == vocab.get_end_token():
            break
        if token == vocab.get_start_token() or token == ',':
            continue

        tokens.append(token)

    return tokens


def generate_visualization_object(dataset, predictions, targets):
    vis_obj = dict()

    vis_obj["predictions"] = predictions
    vis_obj["targets"] = targets
    vis_obj["targets_filepaths"] = [Path(dataset.data_path, filename).absolute().with_suffix(".png") for filename in dataset.filenames]

    with open(Path("tmp_sequenсes").with_suffix(".pkl"), "wb") as writer:
        pickle.dump(vis_obj, writer, protocol=pickle.HIGHEST_PROTOCOL)