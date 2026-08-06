class ConfigLoader:
    def __init__(self, config_path):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        """
        호출 예시:
            config.get("train.csv_data")
            config.get("dev.txt_data")
        """
        value = self.config

        for k in key.split("."):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    