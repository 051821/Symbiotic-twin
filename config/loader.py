import yaml
from pathlib import Path

class ConfigLoader:
    """
    Loads YAML configuration for SYMBIOTIC-TWIN system.
    """

    def __init__(self, config_path: str = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent / "config.yaml"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        self._config = self._load_yaml()

    def _load_yaml(self):
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    @property
    def config(self):
        return self._config

    def get(self, *keys):
        """
        Access nested config values safely.
        Example:
            config.get("system", "num_rounds")
        """
        value = self._config
        for key in keys:
            value = value.get(key)
            if value is None:
                raise KeyError(f"Key {' -> '.join(keys)} not found in config")
        return value


# Global config instance
config_loader = ConfigLoader()
config = config_loader.config
