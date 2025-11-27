import torch
from functools import partial
from collections import OrderedDict
from typing import Union

class FedServerSCAFFOLD:
    def __init__(self, global_params_dict):
        self.global_params_dict = global_params_dict
        self.c_global = global_params_dict.values()
    
    @torch.no_grad()
    def aggregate(self, logger, device, res_cache):
        """
        res_cache: list of (y_delta_list, weights, c_delta_list)
        """
        y_delta_cache = list(zip(*res_cache))[0] # tuple of params_list
        weights_cache = list(zip(*res_cache))[1] # tuple of number_of_samples
        c_delta_cache = list(zip(*res_cache))[2] # tuple of c_delta

        weight_sum = sum(weights_cache) # total number of samples
        weights = torch.tensor(weights_cache, device=device) / weight_sum # 각 client에 곱해줄 가중치 (ratio of number of samples)

        # trainable_parameter = filter(
        #     lambda p: p.requires_grad, self.global_params_dict.values()
        # )

        # # update global model
        # avg_weight = torch.tensor(
        #     [
        #         1 / len(self.clients)
        #         for _ in range(len(self.clients))
        #     ],
        #     device=self.device,
        # )
        
        aggregated_params = []
        for global_param, y_del in zip(self.global_params_dict.values(), zip(*y_delta_cache)): # zip(*updated_params_cache)은 각 개별 parameter별로 client들의 parameter를 묶는다.
            # 즉, 여기서 params는 모든 client들의 같은 파라미터들이다.
            aggregated_params.append(
                global_param + torch.sum(weights * torch.stack(y_del, dim=-1), dim=-1) # 결과의 shape은 원래 parameter의 shape과 같다.
            )
        
        self.global_params_dict = OrderedDict(
            zip(self.global_params_dict.keys(), aggregated_params)
        )

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            # c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_del = torch.sum(weights * torch.stack(c_del, dim=-1), dim=-1)
            
            # c_delta = self.cfg.participation_rate * c_del
            c_delta = c_del / 3
            c_g.add_(c_delta.to(c_g.dtype))   # 또는 .type_as(global_param)

        logger.info("\n>>> Fed Server: Weights are aggregated.\n")