from src.utils.config import load_config
from src.data.loaders import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.trainer_sft import build_trainer

def main():
    config = load_config("configs/debug.yaml")
    print("Config loaded.")

    train_file = config["data"]["train_file"]
    data = load_data(train_file)
    print("Data loaded.")

    processed_data = preprocess_data(data, config)
    print("Preprocess done.")

    trainer = build_trainer(processed_data, config)
    print("Trainer initialized.")

    trainer.train()
    print("Pipeline finished.")

if __name__ == "__main__":
    main()