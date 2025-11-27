import torch
from functools import partial

class FedServer:
    def __init__(self, aggregation_method="fedavg"):
        self.aggregation_method = aggregation_method
        self.aggregation = None
    
    def aggregate(self, logger, weights, num_samples=None, loss_buffer=None):
        if self.aggregation_method == "fedavg":
            self.aggregation = partial(self.fed_avg, num_samples=num_samples)
        elif self.aggregation_method == "fed_w_avg_softmax":
            self.aggregation = partial(self.fed_w_avg_softmax, loss_buffer=loss_buffer)
        elif self.aggregation_method == "fed_w_avg_softmax_scaled":
            self.aggregation = partial(self.fed_w_avg_softmax_scaled, loss_buffer=loss_buffer)
        elif self.aggregation_method == "fed_w_avg_sim":
            self.aggregation = partial(self.fed_w_avg_sim, loss_buffer=loss_buffer)
        elif self.aggregation_method == "fedbn":
            self.aggregation = partial(self.fedbn, num_samples=num_samples)
        assert self.aggregation is not None, "You Should Define Aggregation Method First!"
        
        w_glob_client = self.aggregation(weights) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
        logger.info("\n>>> Fed Server: Weights are aggregated.\n")
        
        return w_glob_client

    # Federated averaging: FedAvg
    def fed_avg(self, weights: list, num_samples: list = None):
        if num_samples is None:
            num_samples = [1 for _ in range(len(weights))]
        # print(f"num_samples = {num_samples}")
        w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
        
        for k in w_avg.keys():
            for i in range(len(weights)):
                w_avg[k] += weights[i][k].detach() * num_samples[i]
            w_avg[k] = torch.div(w_avg[k], sum(num_samples))
        return w_avg
    
    def fedbn(self, weights: list, num_samples: list = None):
        if num_samples is None:
            num_samples = [1 for _ in range(len(weights))]
        
        w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화

        for key in weights[0].keys():
            if 'bn' not in key:
                # temp = torch.zeros_like(weights[0][key], dtype=torch.float32)
                for i in range(len(weights)):
                    w_avg[key] +=  weights[i][key].detach() * num_samples[i]
                w_avg[key] = torch.div(w_avg[key], sum(num_samples))
        
        return w_avg

    # --------------------------------------------------------------------------------------------------------

    # Federated averaging: Federated Weighted Avearge by softmax
    def fed_w_avg_softmax(self, weights, loss_buffer):
        w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()} # OrderedDict에서 각 텐서를 사용하여 초기화
        loss_tensor = torch.tensor(loss_buffer)
                
        # ----------- fed weights by softmax ------------------
        # inverse loss value & graph detach
        loss_inverse = -loss_tensor.detach()
        fed_weights = torch.softmax(loss_inverse, dim=0)
        # -----------------------------------------------------
        
        for k in w_avg.keys():
            for i in range(len(weights)): # client의 개수 - 1 만큼 반복
                w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
        return w_avg

    # Federated averaging: Federated Weighted Avearge by softmax with scaling
    def fed_w_avg_softmax_scaled(self, weights, loss_buffer):
        w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
        
        loss_tensor = torch.tensor(loss_buffer)
                
        # ----------- fed weights by softmax with scale ------------------
        scaling_factor = 10.0
        # inverse loss value & graph detach
        loss_inverse = -loss_tensor.detach() * scaling_factor
        # loss_inverse = torch.log(loss_tensor + 1e-5)
        fed_weights = torch.softmax(loss_inverse, dim=0)
        # -----------------------------------------------------
        
        for k in w_avg.keys():
            for i in range(len(weights)): # client의 개수 - 1 만큼 반복
                w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
        return w_avg

    # Federated averaging: Federated Weighted Avearge by simple average
    def fed_w_avg_sim(self, weights, loss_buffer):
        w_avg = {k: torch.zeros_like(v) for k, v in weights[0].items()}  # OrderedDict에서 각 텐서를 사용하여 초기화
        
        # ----------- fed weights by simple avg ------------------
        loss_tensor = torch.tensor(loss_buffer)
        loss_inverse = 1 / (loss_tensor + 1e-12)
        loss_sum = sum(loss_inverse)
        fed_weights = loss_inverse / loss_sum
        # -----------------------------------------------------
        
        for k in w_avg.keys():
            for i in range(len(weights)): # client의 개수 - 1 만큼 반복
                w_avg[k] += weights[i][k].detach() * fed_weights[i].item()
        return w_avg