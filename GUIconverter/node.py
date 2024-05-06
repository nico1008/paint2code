class Node:
    def __init__(self, key: str, parent_node: 'Node', content_holder: str):
        """
        Initialize a new Node.

        Args:
        key (str): Unique identifier for the node.
        parent_node (Node): Reference to the parent Node.
        content_holder (str): A placeholder string that children's content replaces during rendering.
        """
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child: 'Node'):
        """
        Adds a child Node to the current node's children.

        Args:
        child (Node): The child node to be added.
        """
        self.children.append(child)

    def show(self):
        """Recursively prints the structure of the node and its children."""
        print(f'Node key: {self.key}')
        for child in self.children:
            child.show()

    def render(self, mapping: dict, rendering_function=None):
        """
        Recursively render the content of the node and its children.

        Args:
        mapping (dict): A dictionary with keys mapping to values used for rendering.
        rendering_function (callable, optional): A function applied to transform each node's value.

        Returns:
        str: The rendered content string.

        Raises:
        ValueError: If the key is not found in the mapping or if any child node fails to render.
        """
        content = ""
        for child in self.children:
            placeholder = child.render(mapping, rendering_function)
            if placeholder is None:
                raise ValueError(f"Error rendering child with key {child.key}")
            content += placeholder
            
        value = mapping.get(self.key)
        if value is None:
            raise ValueError(f'The key "{self.key}" could not be found in the mapping.')
        
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if self.children:
            value = value.replace(self.content_holder, content)

        return value

    def __str__(self):
        """Returns a string representation of the node."""
        return f'Node(key: {self.key}, children: {len(self.children)})'
