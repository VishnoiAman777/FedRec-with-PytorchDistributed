# DS603: Privacy-Preserving Recommendation Model

# Overall Report
[Final Report for the paper](Final_Report.pdf)

[DataSet Link and Description](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

## Author: Aman Vishnoi


## Replicating the codebase

The experiments were conducted on 10 nodes AWS-EC2 machine with 9 machines as the client and the last machine as a server. Please put the respective User Data for each of the client in each of the machines.

```
  $ wget https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
  $ bash Anaconda3-5.1.0-Linux-x86-64.sh
  $ source ~/.bashrc
  $ conda env update --file environment.yml --prune
```

Now place the clients and server data in the UserData folder. We have divided the MIND data into 10 equal parts and put each part on each client machine respectively(server is takng part in the training also). 

Now in order to run the application run this command in each of the client machines

```
 $ torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_PROCESSES_PER_NODE
    --rdzv-id=100
    --rdzv-backend=c10d
    --rdzv-endpoint=$SERVER_IP
    client.py 
```

And for the server

```
 $ torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_PROCESSES_PER_NODE
    --rdzv-id=100
    --rdzv-backend=c10d
    --rdzv-endpoint=$SERVER_IP
    server.py 
```

When all the client machines are up and running and the server starts the client and server will start communicating with each other to share the gradients and weights.

### Abstract
The primary objective of this project is to develop a privacy-preserving federated news recommendation system. Traditional news recommendation methods rely on centralized storage of user behavior data for model training, which can raise privacy concerns due to the sensitive nature of user behaviors. In this project, we aim to create a privacy-preserving method for news recommendation model training based on federated learning, where user behavior data remains locally stored on user devices. The project employs the MIND dataset, which contains anonymized behavior logs from the Microsoft News website. Several novel approaches will be explored in this project, including the integration of differential privacy, monitoring model performance against federated attacks, testing the model on new benchmarks, such as song datasets, and implementing fairness-aware Federated Matrix Factorization.

<img src="https://drive.google.com/uc?id=1Vnk1L1rDaL0Or67VUOm9rylSOj01aVQH">

## I. Introduction
Many existing news recommendation methods still rely on centralized storage of user behavior data for model training, which raises privacy concerns. To address this issue, we propose a federated learning approach where user data is kept locally on their devices. This approach leverages the collective user data to train accurate recommendation models without the need for centralized storage. The project involves maintaining a small user model on each edge device and sending gradients to a central server for training a global news model. Techniques such as Multiparty computation and Local Differential Privacy are used for privacy protection. The updated global model is then distributed back to each user device for local model updates. This process is repeated over multiple rounds. The project also monitors model performance against various attacks, including substitution-based profile pollution attacks and model poisoning. Additionally, fairness and robustness considerations are incorporated into the federated learning system.

## II. Related Work
Previous work in federated recommendation systems has been conducted by researchers such as Tao Qi, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, and Xing Xie. However, their approach differs from ours as they utilize multiparty computation instead of differential privacy or homomorphic encryption. Moreover, their model has not been tested against attacks and lacks fairness considerations. Another approach by Tao Qi, Fangzhao Wu, Chuhan Wu, Yongfeng Huang, and Xing Xie addresses privacy-preserving news recommendation model learning but incurs high communication and computation costs due to the large size of news recommendation models.

## III. System Model
The project utilizes the MIND dataset for news articles and processes data on a system with a P5000 GPU, 30 GB RAM, and a 16-core CPU. The experiment logs have been published using WandB. Data from the "addressa" dataset is preprocessed into the MIND dataset format to be used in the model.

## IV. Algorithms
The project begins by formatting the news dataset. User histories, clicked news, and ignored news are categorized as positive and negative samples. A dataloader is created to aggregate data from randomly selected 50 users, and Multiparty computation is employed for aggregation. A pretrained BERT model is used to generate news vector embeddings. User-encoder models are present at both the server and client devices. The user model incorporates multihead attention layers connected to linear layers to generate user embeddings. The project uses collaborative filtering and cross-entropy loss to train the network. The user model parameters are updated and distributed across the client devices.

## V. Experiments and Results
The primary datasets used in the project are the MIND dataset and the "addressa" dataset. The results for the MIND dataset are as follows:

**MIND Dataset:**
- MRR: 32.86
- AUC: 68.42
- NDCG@5: 36.43
- NDCG@10: 42.62

**Addressa Dataset:**
- MRR: 37.67
- AUC: 72.04
- NDCG@5: 37.33
- NDCG@10: 45.31

## VI. Enhancements Done in the Paper
The project's code has been developed from scratch with the assistance of the original repository. Several enhancements have been made to the client architecture, including the use of two models: user-encoder and text-encoder. The text-encoder employs the Distill-Bert architecture to reduce parameters by 40%. To adapt the text encoder to each user, a neural architecture has been added in front of Distill Bert. The user encoder takes user history encoding and generates a 400-dimensional embedding. The project incorporates differential privacy by adding Gaussian noise to gradients during mini-batch processing. The code optimizes memory usage and handles edge cases.

## VII. Implementation via Torch Distributed
The project code has been adapted for compatibility with PyTorch Distributed. Five EC2 t2-medium instances with 2-core CPUs and 4GB memory are used, with four acting as clients and one as a server. The server aggregates parameters from clients. Various training strategies, including parameter averaging and gradient averaging, have been explored. The project employs the Slurm backend task scheduler to manage resource allocation. However, challenges related to client failure, high communication costs, and model irregularities have been encountered.

## VIII. Issues in the Paper
Several issues in the paper need to be addressed, including uncertainties regarding how the user model at the client-end is used, lack of clarity on news vector generation, unexplained calculation of communication overhead reduction, inconsistencies in scaling training loss, difficulties replicating results, and concerns about user privacy in multiparty computation.

## IX. Conclusion
This project explores the development of a privacy-preserving federated news recommendation system using the MIND dataset. Various techniques, including differential privacy, have been implemented to protect user privacy. The project also monitors model performance against attacks and incorporates fairness considerations. Future work involves curating datasets, investigating the impact of differential privacy, implementing PyTorch Distributed on real clients, and addressing issues related to client failure and communication costs.


