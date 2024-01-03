class ConfigClass:
    def __init__(self, config_dict):
        self.deepscale = False
        self.fp16 = False
        self.no_cuda = False
        self.local_rank = None
        for key, value in config_dict.items():
            setattr(self, key, value)

def dict_to_config_class(config_dict):
    return ConfigClass(config_dict)