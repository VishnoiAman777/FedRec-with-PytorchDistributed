import torch
import socket
import json
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
from model import UserModel
import torch.optim as optim
from evaluation_functions import roc_auc_score, mrr_score, ndcg_score
import wandb
from datetime import datetime, timedelta
from opacus import PrivacyEngine


"""
This function enables us to process news gradients
"""
def process_news_grad(candidate_info, his_info):
    news_grad = {}
    candidate_news, candidate_vecs, candidate_grad = candidate_info
    his, his_vecs, his_grad = his_info

    candidate_news, candaidate_grad = (
        candidate_news.reshape(-1, ),
        candidate_grad.reshape(-1, 400),
    )
    his, his_grad = his.reshape(-1, ), his_grad.reshape(-1, 400)

    for nid, grad in zip(his, his_grad):
        if nid in news_grad:
            news_grad[nid] += grad
        else:
            news_grad[nid] = grad

    for nid, grad in zip(candidate_news, candaidate_grad):
        if nid in news_grad:
            news_grad[nid] += grad
        else:
            news_grad[nid] = grad
    return news_grad


"""
This function allows us to process user-gradients i.e. gradients for the user encoder
"""
def process_user_grad(model_param):
    user_grad = {}
    for name, param in model_param:
        user_grad[name] = param.grad
    return user_grad


def train_on_step(model, train_dl, optimizer, dp_enabled, sigma):
    # These are generating news vectors from transformers for given nids

    model.train()
    loss = 0

    for _, batch_sample in enumerate(train_dl):
        # model.user_encoder.load_state_dict(agg.user_encoder.state_dict())
        candidate_news, his, label = batch_sample
        bz_loss, y_hat, candidate_news_vecs, his_vecs = model(candidate_news, his, label)
        loss += bz_loss.detach().cpu().numpy()

        print("The loss we have encountered is", loss)

        optimizer.zero_grad()
        bz_loss.backward()

        candaidate_grad = candidate_news_vecs.grad.detach().cpu() 

        candidate_vecs = candidate_news_vecs.detach().cpu().numpy()
        candidate_news = candidate_news.numpy()

        his_grad = his_vecs.grad.detach().cpu() 
        his_vecs = his_vecs.detach().cpu().numpy()
        his = his.numpy()
                
        if dp_enabled:
            candaidate_grad += torch.normal(mean=0.0, std=sigma, size=candidate_news_vecs.shape)
            his_grad += torch.normal(mean=0.0, std=sigma, size=his_vecs.shape)
        
        news_grad = process_news_grad(
            [candidate_news, candidate_vecs, candaidate_grad], [his, his_vecs, his_grad]
        )
        # Return a news grad: a dic that contains nid: gradient_curresponding to that news
        user_grad = process_user_grad(
            model.user_encoder.named_parameters()
        )
        model.collect(news_grad, user_grad)

    model.update()
    return loss



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        isTrain = True
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        # self.model = model.to(self.local_rank)
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        if isTrain:
            self.model = DDP(self.model)
        else:
            self.model = model

    def _load_snapshot(self, snapshot_path):
        # loc = f"cuda:{self.local_rank}"
        # snapshot = torch.load(snapshot_path, map_location=loc)
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path} \n")

    def validate(self, valid_sam, nid2index, news_index, device):
        # agg.gen_news_vecs(list(range(len(news_index))))
        # model.user_encoder.eval()
        self.model.eval()
        roc = []
        mrr = []
        ndgc5 = []
        ndgc10 = []
        v_loss = []
        for i in valid_sam:
            his = torch.Tensor([[nid2index[j] for j in i[3]] + [0]*(50-len(i[3]))]).to(torch.int)
            candidate_news = torch.Tensor([[nid2index[i[1]]] + [nid2index[j] for j in i[2][-4:]]]).to(torch.int)
            label = torch.Tensor([0]).to(torch.long)
            loss, y_hat, _, _ = self.model(candidate_news, his, label)
            loss = loss.detach().numpy()
            score = y_hat.detach().numpy()[0]
            label = torch.Tensor([1, 0, 0, 0, 0]).to(torch.int).detach().numpy()
            roc.append(float(roc_auc_score(label, score)))
            mrr.append(float(mrr_score(label, score)))
            ndgc5.append(float(ndcg_score(label, score, 5)))
            ndgc10.append(float(ndcg_score(label, score, 10)))
            v_loss.append(loss)
        return sum(v_loss)/len(v_loss), round(float(roc_auc_score(label, score)), 2), round(float(mrr_score(label, score)), 2), round(float(ndcg_score(label, score, 5)), 2), round(float(ndcg_score(label, score, 10)), 2)
    

    def train(self, max_epochs: int, valid_sam, nid2index, news_index, device, dp_enabled, sigma=None):
        for epoch in range(self.epochs_run, max_epochs):
            loss = train_on_step(self.model,  self.train_data, self.optimizer, dp_enabled, sigma)
            print("Completed Epoch ", epoch, "with loss", loss)
            # if epoch % self.save_every == 0:
            #     self._save_snapshot(epoch)
            val_loss, val_auc, val_mrr, val_ndcg, val_ndcg10 = self.validate(valid_sam, nid2index, news_index, device)
            print("The validation metrics obtained from the above graph are",val_loss, val_auc, val_mrr, val_ndcg, val_ndcg10 )
            wandb.log({
                "training_loss": loss,
                "validation_loss":val_loss,
                "valid_auc": val_auc,
                "valid_mrr": val_mrr,
                "val_ndcg@5": val_ndcg,
                "val_ndcg@10": val_ndcg10
            })
            
def send_saved_model():
    # Sending the model state_dict to the server
    host = "172.31.40.109"
    port = 12345
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Trying to connect")
    s.connect((host, port))
    print("Connection Successfull")
    model_file = open('model.pt', 'rb')
    print("Started Sending Data")
    while True:
        data = model_file.read(1024)
        if not data:
            break
        s.sendall(data)

    # Close the file and socket
    model_file.close()
    s.close()   
    print("Sending COmplete")
    
            
def main(save_every: int, total_epochs: int, batch_size: int, dp_Epsilon:int, dp_Enabled, name, snapshot_path: str = "snapshot.pt"):
    wandb.login(key="0b610249246639a266b453e749037617d05cca31")
    wandb.init(
        mode="disabled",
        project="Node4",
        name=name
    )
    MAX_GRAD_NORM = 2     # Equivalent to the clipping threshold C
    EPSILON = dp_Epsilon
    DELTA = 1e-5
    EPOCHS = 50
    LR = 0.00005
    privacy_engine = PrivacyEngine()

    init_process_group(backend="gloo", init_method="env://", timeout=timedelta(days = 2)) # Setting up gloo backend for distributed training

    data_path = Path("UserData")
    # Loading all the data for our User
    with open(data_path / "bert_nid2index.pkl", "rb") as f:
        nid2index = pickle.load(f)

    news_index = np.load(data_path / "bert_news_index.npy", allow_pickle=True)

    with open(data_path / "train_sam_uid.pkl", "rb") as f:
        train_sam = pickle.load(f)

    with open(data_path / "valid_sam_uid.pkl", "rb") as f:
        valid_sam = pickle.load(f)

    """ 
        Train_ds returns
        candidate_news: Integer currsponding to news watched by one user sampled from set of users, 
        his: History of user 
        label: 0
    """
    train_ds = TrainDataset(train_sam, nid2index, news_index)
    train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_ds))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    news_dataset = NewsDataset(news_index)
    model = UserModel(news_dataset, news_index, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.00005)

    t = torch.tensor(2.0)
    dist.broadcast(t, src=1)
    while int(t) != 0: 
        print("Starting model_sync")
        # Syncronization of model from server
        for index, param in enumerate(model.parameters()):
            data_rcvd = torch.zeros_like(param.data)
            dist.broadcast(data_rcvd, src=1)
            param.data = data_rcvd
        
        print("Model Sync Successfull")

                
        sigma = None
        if dp_enabled:
            final_model, final_optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dl,
                epochs=EPOCHS,
                target_epsilon=EPSILON,
                target_delta=DELTA,
                max_grad_norm=MAX_GRAD_NORM,
            )
            sigma = final_optimizer.noise_multiplier
            print("Applying sigma", sigma, "for DP")

        trainer = Trainer(model, train_dl, optimizer, save_every, snapshot_path, False)
        trainer.train(total_epochs, valid_sam, nid2index, news_index, device, dp_enabled, sigma)
        print("Completed the training")
        model_dict = model.state_dict()
        print("Model_State_dict_is_obtained")
        torch.save(model_dict, "model.pt")
        print("Model Has been saved successfully")
        # This function is to send the saved model to the server
        send_saved_model()
        dist.broadcast(t, src=1)

    destroy_process_group()


if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    save_every = int(sys.argv[3])
    dp_Epsilon = int(sys.argv[4])
    dp_enabled = bool(dp_Epsilon)
    name=sys.argv[5]
    
    # init_process_group(backend="gloo") # Setting up gloo backend for distributed training
    # print(int(os.environ["LOCAL_RANK"]))
    # print("$"*100)

    # dist.send(torch.tensor([1, 2]), dst=0)
    main(save_every, total_epochs, batch_size, dp_Epsilon, dp_enabled, name)
    
    
    # destroy_process_group()
