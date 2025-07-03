from creating_dataset import create_dataset
from train_model import train_with_dataset
import torch.multiprocessing as mp

if __name__ == "__main__":
    NUMBER_OF_TRAINING = 1
    mp.set_start_method("spawn", force=True)

    for _ in range(NUMBER_OF_TRAINING):
        create_dataset(model="current_net_2.pth.tar", number_of_proces=15, number_of_games_per_process=20)
        train_with_dataset(model_name="current_net_2.pth.tar",save_model_name="current_net_3.pth.tar")

