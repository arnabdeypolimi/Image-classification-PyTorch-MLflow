import json

from modules.utils.utils import str2bool


class ConfigurationManager:
    """
    The configuration manager class is responsible for reading, parsing and accessing configuration files.
    """

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_file(cls, name):
        if not name.endswith('.json'):
            name = name + ".json"

        with open(f'config_files/{name}') as config_file:
            return cls(json.load(config_file))

    @property
    def model_name(self):
        return self.config.get('model_name', '')

    @property
    def num_classes(self):
        # todo: use some library to take into account different ways to write a country language
        return int(self.config.get('num_classes', "17"))

    @property
    def feature_extract(self):
        return str2bool(self.config.get('feature_extract', "False"))

    @property
    def pre_trained(self):
        return str2bool(self.config.get('pre_trained', "False"))

    @property
    def batch_size(self):
        return int(self.config.get('batch_size', "32"))

    @property
    def num_epochs(self):
        return int(self.config.get('num_epochs', "50"))

    @property
    def save_model(self):
        return self.config.get('save_model', '')

    @property
    def save_confusion_mat(self):
        return self.config.get('save_confusion_mat', '')