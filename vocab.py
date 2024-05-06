import json

START_TOKEN = "START"
END_TOKEN = "END"
PADDING_TOKEN = "PADDING"
UNKNOWN_TOKEN = "UNKNOWN"

class Vocab:
    def __init__(self, vocab_path=None):
        """Initialize the vocabulary with predefined tokens and optionally load additional tokens from a file."""
        init_tokens = [START_TOKEN, END_TOKEN, PADDING_TOKEN, UNKNOWN_TOKEN]
        self.token_to_id = {token: i for i, token in enumerate(init_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(init_tokens)}

        if vocab_path:
            self.__read_vocab_from_file(vocab_path)

    def add_token(self, token):
        """Add a token to the vocabulary if it is not already present."""
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def __read_vocab_from_file(self, vocab_path):
        """Read tokens from a file and add them to the vocabulary."""
        try:
            with open(vocab_path, "r") as reader:
                tokens = reader.read().split()
                for token in tokens:
                    self.add_token(token)
        except FileNotFoundError:
            print(f"Warning: The file {vocab_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading from {vocab_path}: {e}")

    def get_token_by_id(self, id):
        """Return the token associated with an ID; return UNKNOWN_TOKEN if ID is not found."""
        return self.id_to_token.get(id, self.token_to_id[UNKNOWN_TOKEN])

    def get_id_by_token(self, token):
        """Return the ID associated with a token; return the ID of UNKNOWN_TOKEN if token is not found."""
        return self.token_to_id.get(token, self.token_to_id[UNKNOWN_TOKEN])

    def get_start_token(self):
        """Return the start token."""
        return START_TOKEN

    def get_end_token(self):
        """Return the end token."""
        return END_TOKEN

    def get_padding_token(self):
        """Return the padding token."""
        return PADDING_TOKEN

    def __str__(self):
        """Return the string representation of the vocabulary."""
        return json.dumps(self.token_to_id, indent=2)

    def __len__(self):
        """Return the number of tokens in the vocabulary."""
        return len(self.token_to_id)