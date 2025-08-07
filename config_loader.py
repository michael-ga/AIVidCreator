import json
import os

CONFIG_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config_video_creation.json')

def load_config(config_path=None):
    """
    Loads a JSON config file and returns the config as a dictionary.
    If config_path is None, loads the default config file from the project root.
    Args:
        config_path (str): Path to the config JSON file.
    Returns:
        dict: Configuration dictionary.
    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if config_path is None:
        config_path = CONFIG_DEFAULT_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

class VideoConfig:
    def __init__(self, config_path=None):
        config = load_config(config_path)
        self.effects_path = config.get('effects_path')
        self.audio_path = config.get('audio_path')
        self.content_path = config.get('content_path')
        self.content_workitems = config.get('content_workitems')

    def __repr__(self):
        return f"<VideoConfig effects_path={self.effects_path} audio_path={self.audio_path}>"
