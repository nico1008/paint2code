from pathlib import Path
from PIL import Image
import torch

class Paint2CodeDataset:
    def __init__(self, data_path, split, vocab, transform=None):
        
        """Initialize the dataset with the path to the data,
        the dataset split, the vocabulary,
        and optional transformations.
        """
        
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split name '{split}'. Expected one of: 'train', 'validation', 'test'.")

        self.data_path = Path(data_path)
        self.transform = transform
        self.vocab = vocab

        dataset_file = self.data_path.parent / f'{split}_dataset.txt'
        if not dataset_file.exists():
            raise FileNotFoundError(f"The dataset file {dataset_file} does not exist.")

        with dataset_file.open("r") as file:
            self.filenames = [line.strip() for line in file if line.strip()]
            

    def __len__(self):
        
        """Return the number of items in the dataset."""
        
        return len(self.filenames)
    

    def __getitem__(self, idx):
        
        """Retrieve an image and its corresponding token IDs by index."""
        
        img_path = self.data_path / f"{self.filenames[idx]}.png"
        tokens_path = self.data_path / f"{self.filenames[idx]}.gui"

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        tokens = self.parse_gui_token_file(tokens_path)
        tokens = [self.vocab.get_start_token()] + tokens + [self.vocab.get_end_token()]

        token_ids = [self.vocab.get_id_by_token(token) for token in tokens]
        token_ids = torch.tensor(token_ids, dtype=torch.long)  # Use torch.tensor directly with the proper type

        return image, token_ids
    

    def parse_gui_token_file(self, filepath):
        
        """Parse the GUI token file and return a list of tokens."""
        
        if filepath.suffix != ".gui":
            raise ValueError(f"Expected a '.gui' file, got {filepath.suffix} instead.")

        with filepath.open("r") as file:
            raw_data = file.read().replace(', ', ' , ')
            tokens = [token for token in raw_data.split() if token]  # This avoids empty strings

        return tokens
