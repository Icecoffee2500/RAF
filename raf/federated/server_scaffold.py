import torch
from functools import partial

class FedServerSCAFFOLD:
    def __init__(self):
        self.aggregation = None
        self.global_params_dict = None
        self.c_global = None
    
    def aggregate(self, logger, weights, num_samples=None, loss_buffer=None):
        if self.aggregation_method == "fed_avg":
            self.aggregation = partial(self.fed_avg, num_samples=num_samples)
        elif self.aggregation_method == "fed_w_avg_softmax":
            self.aggregation = partial(self.fed_w_avg_softmax, loss_buffer=loss_buffer)
        elif self.aggregation_method == "fed_w_avg_softmax_scaled":
            self.aggregation = partial(self.fed_w_avg_softmax_scaled, loss_buffer=loss_buffer)
        elif self.aggregation_method == "fed_w_avg_sim":
            self.aggregation = partial(self.fed_w_avg_sim, loss_buffer=loss_buffer)
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