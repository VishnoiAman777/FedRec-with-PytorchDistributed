import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
from dataset import TrainDataset, NewsDataset
from pathlib import Path
from model import UserModel
import torch.optim as optim



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


def train_on_step(model, train_dl, optimizer):
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
        
        news_grad = process_news_grad(
            [candidate_news, candidate_vecs, candaidate_grad], [his, his_vecs, his_grad]
        )
        # Return a news grad: a dic that contains nid: gradient_curresponding to that news
        user_grad = process_user_grad(
            model.module.user_encoder.named_parameters()
        )
        model.module.collect(news_grad, user_grad)

    model.module.update()
    return loss



class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
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

        self.model = DDP(self.model)

    def _load_snapshot(self, snapshot_path):
        # loc = f"cuda:{self.local_rank}"
        # snapshot = torch.load(snapshot_path, map_location=loc)
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path} \n")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_on_step(self.model,  self.train_data, self.optimizer)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)



def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    init_process_group(backend="gloo") # Setting up gloo backend for distributed training

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
    trainer = Trainer(model, train_dl, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    save_every = int(sys.argv[3])
    main(save_every, total_epochs, batch_size)