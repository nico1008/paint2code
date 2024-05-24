import string
import random


class Utils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        """Generates random text of a specified length, with optional spaces and upper case.

        Args:
            length_text (int): Length of the text to generate.
            space_number (int): Number of spaces to insert into the text.
            with_upper_case (bool): If True, capitalize the first letter.

        Returns:
            str: The generated random text.
        """
        text = ''.join(random.choices(string.ascii_letters, k=length_text))

        if with_upper_case:
            text = text.capitalize()

        space_positions = random.sample(range(1, len(text)), min(space_number, len(text) - 1))
        for pos in sorted(space_positions):
            text = text[:pos] + ' ' + text[pos:]

        return text

    @staticmethod
    def render_content_with_text(key, value):
        """Replace placeholder in the value based on the key using a predefined text if applicable.

        Args:
            key (str): The type of content key.
            value (str): The string value containing a placeholder to replace.

        Returns:
            str: The modified string with the placeholder replaced if conditions match.
        """
        FILL_WITH_RANDOM_TEXT = True
        TEXT_PLACE_HOLDER = "[]"

        if FILL_WITH_RANDOM_TEXT:
            if "btn" in key:
                value = value.replace(TEXT_PLACE_HOLDER, "Button")
            elif "title" in key:
                value = value.replace(TEXT_PLACE_HOLDER, "Title")
            elif "text" in key:
                value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=64, space_number=9, with_upper_case=False))

        return value