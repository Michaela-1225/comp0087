class DummyTrainer:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def train(self):
        print("Dummy training started.")
        print(f"Training samples: {len(self.data)}")
        print("Dummy training finished.")

def build_trainer(data, config):
    return DummyTrainer(data, config)