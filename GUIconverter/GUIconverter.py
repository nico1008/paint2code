import json
from .node import Node
from .utils import Utils
from pathlib import Path

#paint2code transpiler  
class GUIconverter:
    def __init__(self, style='style1'):
        """
        Initialize the GUI converter with a specified style.

        Args:
        style (str): The name of the style to load for the GUI.
        """
        # Path to the styles directory relative to this file
        base_path = Path(__file__).parent / "styles"
        style_file_path = base_path / f"{style}.json"
        
        # Load the style configuration from the specified JSON file
        with style_file_path.open() as data_file:
            self.dsl_mapping = json.load(data_file)

        # Initialize properties based on loaded style
        self.opening_tag = self.dsl_mapping.get("opening-tag", "")
        self.closing_tag = self.dsl_mapping.get("closing-tag", "")
        self.content_holder = self.opening_tag + self.closing_tag

        # Root node always starts as 'body'
        self.root = Node("body", None, self.content_holder)

    def transpile(self, tokens, output_file_path=None, insert_random_text=False):
        """
        Converts an array of tokens into HTML based on the loaded style and writes it to a file if specified.

        Args:
        tokens (list): List of tokens to be converted.
        output_file_path (str, optional): File path to write the resulting HTML.
        insert_random_text (bool, optional): If true, random text is inserted into the content.

        Returns:
        str: The resulting HTML content.

        Raises:
        ValueError: If the tokens list is empty.
        """
        if not tokens:
            raise ValueError('Tokens must be a non-empty array')

        rendering_function = Utils.render_content_with_text if insert_random_text else None

        current_parent = self.root
        last_inserted_element = None

        for token in tokens:
            if token == self.opening_tag:
                current_parent = last_inserted_element
            elif token == self.closing_tag:
                current_parent = current_parent.parent if current_parent.parent else self.root
            else:
                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                last_inserted_element = element

        output_html = self.root.render(self.dsl_mapping, rendering_function)

        if output_file_path:
            with open(output_file_path, 'w') as output_file:
                output_file.write(output_html)

        return output_html
