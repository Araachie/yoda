import yaml

from lutils.dict_wrapper import DictWrapper


class Configuration(DictWrapper):
    """
    Represents the configuration parameters for running the process
    """

    def __init__(self, path: str):
        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super(Configuration, self).__init__(config)

        self.check_config()

    def check_config(self):
        if "with_flows" not in self["data"]:
            self["data"]["with_flows"] = False

        if "skip_prob" not in self["model"]:
            self["model"]["skip_prob"] = 0.5
