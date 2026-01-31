import torch.nn.functional as F
import torch

def calculate_SPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)
    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)
    
    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def calculate_MMD_loss(human_crit, sample_crit):
    mmd_loss = human_crit.mean() - sample_crit.mean()
    return mmd_loss

def calculate_reconstruct_loss1(original_crit, sample_crit):
    mmd_loss = original_crit - sample_crit
    # mmd_loss = original_crit - sample_crit / (torch.square(sample_crit) + torch.square(original_crit)).sqrt()
    # mmd_loss = -torch.square(original_crit - sample_crit) / (torch.square(sample_crit) + torch.square(original_crit))
    return mmd_loss

def calculate_reconstruct_loss2(original_crit, sample_crit):
    # mmd_loss = sample_crit - original_crit / (sample_crit + original_crit)
    mmd_loss = sample_crit - original_crit
    return mmd_loss