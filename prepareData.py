import logging
from pathlib import Path
from math import floor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths directly
data_path = Path("data/all_data")
vocab_output_path = data_path.parent / "vocab.txt"

# Set the dataset splits percentages
TRAIN_PER = 0.8
TEST_PER = 0.1
VAL_PER = 0.1

def count_file_types(directory):
    """Count occurrences of each file type within the directory."""
    occurrences = {}
    for file in directory.iterdir():
        if file.is_file():  # Ensuring to count only files
            occurrences[file.stem] = occurrences.get(file.stem, {})
            occurrences[file.stem][file.suffix] = occurrences[file.stem].get(file.suffix, 0) + 1
    return occurrences

def get_valid_pairs(occurrences):
    """Filter for valid pairs of .gui and .png files."""
    return [key for key, value in occurrences.items() if value.get(".gui", 0) == 1 and value.get(".png", 0) == 1]

def save_data_splits(data_splits, parent_directory):
    """Save data splits to files."""
    try:
        for key, value in data_splits.items():
            filepath = parent_directory / f'{key}_dataset.txt'
            with open(filepath, "w") as writer:
                for example in value:
                    writer.write(example + "\n")
    except IOError as e:
        logging.error(f"Error writing to file: {e}")

def main():
    occurrences_count = count_file_types(data_path)
    valid_pairs = get_valid_pairs(occurrences_count)
    number_of_examples = len(valid_pairs)

    # Compute split indices
    train_split = floor(number_of_examples * TRAIN_PER)
    validation_split = floor(number_of_examples * VAL_PER)

    # Create datasets
    train_set = valid_pairs[:train_split]
    validation_set = valid_pairs[train_split:train_split + validation_split]
    test_set = valid_pairs[train_split + validation_split:]

    # Save dataset splits to files
    dataset_splits = {"train": train_set, "validation": validation_set, "test": test_set}
    save_data_splits(dataset_splits, data_path.parent)

    # Initialize a set to keep unique tokens
    all_tokens = set()
    # Iterate through .gui files to extract tokens
    for file in data_path.glob("*.gui"):
        if file.is_file():  # Ensuring only files are processed
            with open(file, "r") as reader:
                tokens = reader.read().replace('\n', ' ').replace(', ', ' , ').split()
                all_tokens.update(tokens)

    # Write the set of all tokens to a vocab file
    try:
        with open(vocab_output_path, "w") as writer:
            writer.write(" ".join(sorted(all_tokens)))
    except IOError as e:
        logging.error(f"Error writing to vocab file: {e}")

    logging.info(f'Found a total of {number_of_examples} valid examples')
    logging.info(f'Writing vocab with {len(all_tokens)} tokens to {vocab_output_path}')

if __name__ == "__main__":
    main()