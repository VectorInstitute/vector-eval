import json
import shutil
import yaml


def load_from_yaml(file_path):
    """Loads content from a YAML file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

def write_to_json(data, file_path):
    """Writes a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_from_json(file_path):
    """Reads content from a JSON file and returns a dictionary."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def delete_directory(directory_path):
    """Deletes a directory and its contents."""
    shutil.rmtree(directory_path)