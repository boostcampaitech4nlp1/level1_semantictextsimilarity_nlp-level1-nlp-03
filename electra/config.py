import json


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        self.__dict__.update(data)
        

    @property
    def dict(self):
        return self.__dict__

        
if __name__ == "__main__":
    config_path = "./config.json"
    config = Config(config_path)
    print(config.model_name)
    print(config.batch_size)
    config.batch_size
