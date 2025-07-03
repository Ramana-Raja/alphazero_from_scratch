import os
import pickle
from model_components.alpha_net import ChessNet,train
import torch

def train_with_dataset(dataset_loc="./model_components/datasets",model_name="",save_model_name=""):

    data_path = dataset_loc
    dataset_for_train = []
    for filen in os.listdir(data_path):
        fodlername = os.path.join(data_path, filen)
        for file in os.listdir(fodlername):
            filename = os.path.join(fodlername,file)
            with open(filename, 'rb') as fo:
                dataset_for_train.extend(pickle.load(fo, encoding='bytes'))
    model = ChessNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if len(model_name) != 0:
        current_net_filename = os.path.join("./model_components/model_data/", model_name)
        checkpoint = torch.load(current_net_filename)
        model.load_state_dict(checkpoint['state_dict'])
        train(model,dataset_for_train)
        torch.save({'state_dict': model.state_dict()}, os.path.join("./model_components/model_data/", save_model_name))
    else:
        train(model,dataset_for_train)
        torch.save({'state_dict': model.state_dict()}, os.path.join("./model_components/model_data/", save_model_name))