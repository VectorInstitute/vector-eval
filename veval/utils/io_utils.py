import yaml


def load_from_yaml(file_path):
    """Loads content from a `yaml` file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data