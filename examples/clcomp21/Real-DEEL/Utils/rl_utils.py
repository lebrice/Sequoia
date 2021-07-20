from torch.distributions import Categorical
import torch 

def sample_action(logits: torch.tensor, return_entropy_log_prob=False):
    prob_dist = Categorical(logits=logits)
    y_preds = prob_dist.sample()
    if return_entropy_log_prob:
        return y_preds, prob_dist.log_prob(y_preds), prob_dist.entropy()
    return y_preds