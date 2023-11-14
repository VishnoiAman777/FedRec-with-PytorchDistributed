"""
    The server will only be used for aggregating the 
    graients and nothing else.
"""
import torch.distributed as dist
import socket
import numpy as np
import threading
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle5 as pickle
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
from dataset import TrainDataset, NewsDataset
from pathlib import Path
from model import ServerUserModel
import torch.optim as optim
from datetime import timedelta

def handle_client(conn, addr, no):
    # Receive the JSON string
    with open(f'received_model_{no}.pt', 'wb') as f:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            f.write(data)
    print(f"Received from {addr}: saved_model.pt")
    # Close the connection
    conn.close()

def average_all_params_rcvd():
    state_dicts = []
    world_size = dist.get_world_size()
    for i in range(1, world_size):
        saved_model_name = f"received_model_{i}.pt"
        state_dicts.append(torch.load(saved_model_name))
    print("Obtained all the state dicts")
    # Compute the average of model parameters
    averaged_params = []
    for tensors in zip(*[sd.values() for sd in state_dicts]):
        # Average tensors element-wise
        summed_tensor = torch.stack(tensors).sum(0)
        averaged_tensor = summed_tensor / (world_size -1)
        averaged_params.append(averaged_tensor)
    
    print("Completed Averaging all the state_dicts")
    state_dict_keys = state_dicts[0].keys()
    averaged_state_dict = {key: param for key, param in zip(state_dict_keys, averaged_params)}
    return averaged_state_dict


def main(global_epochs):
    init_process_group(backend="gloo",init_method="env://", timeout=timedelta(days = 2)) # Setting up gloo backend for distributed training   
    # print("THe rank of the process is", dist.get_rank())
    # float_tensor = torch.tensor([1.0, 1.0, 1.0])
    # dist.broadcast(float_tensor, src=1)
    # print("Tenosr BRoadcasted")    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ServerUserModel(device).to(device)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    epochs = 0
    
    while epochs < global_epochs:
        epochs += 1
        dist.broadcast(torch.tensor(1.0), src=dist.get_rank()) # This tensor is just an indication wheather we want to continue or not
        # Broadcasting the entire model
        for i in model.parameters():
            dist.broadcast(i.data, src=dist.get_rank())
        print("Model Syncronization on clients done")
            
        host = '0.0.0.0'  # Listen on all available network interfaces
        port = 12345  
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(10)
        
        # Start number of threads to handle each client connection
        client_no = 0
        for i in range(world_size-1):
            client_no += 1
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, client_no))
            t.start()

        # Wait for all threads to finish
        for t in threading.enumerate():
            if t != threading.current_thread():
                t.join()
                
        print("Received all models")
        # Now we can average the model received
        averaged_state_dict = average_all_params_rcvd()
        model.load_state_dict(averaged_state_dict)
        
    dist.broadcast(torch.tensor(0.0), src=dist.get_rank()) # This tensor is to tell clients to stop now all global epochs have been completed
    destroy_process_group()

if __name__ == "__main__":
    import sys

    # num_clients = int(sys.argv[1])
    # total_epochs = int(sys.argv[2])
    global_epochs = int(sys.argv[1])
    main(global_epochs)
