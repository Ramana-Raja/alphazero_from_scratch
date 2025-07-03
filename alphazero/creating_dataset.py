from model_components.MCTS import MCTS_self_play
import torch.multiprocessing as mp
from model_components.alpha_net import ChessNet
import torch
import os

def create_dataset(model="current_net_2.pth.tar",number_of_games_per_process=10,number_of_proces=15):
    j=0
    dir_path = "C:\\Users\\raman\\PycharmProjects\\pythonProject1\\pytorch\\alphazero\\model_components\\datasets"
    for file in os.listdir(dir_path):
        j +=1
    print(j)

    MODEL_FROM_TRAINING = model
    NUMBER_OF_GAMES_PER_PROCESS = number_of_games_per_process
    NUMBER_OF_PROCESS = number_of_proces

    model = ChessNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.share_memory()
    current_net_filename = os.path.join("./model_components/model_data/", MODEL_FROM_TRAINING)
    checkpoint = torch.load(current_net_filename)
    model.load_state_dict(checkpoint['state_dict'])

    processes = []

    for i in range(NUMBER_OF_PROCESS):
        p = mp.Process(target=MCTS_self_play, args=(model, NUMBER_OF_GAMES_PER_PROCESS, i,j))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()