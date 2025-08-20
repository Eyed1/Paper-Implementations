import torch as t
from torch import nn
import einops

def lower_t_loss(true_eps, pred_eps, beta_t, alpha_t, alpha_prod_t):
    sigma_t2 = beta_t * (1 - (alpha_prod_t)/alpha_t)/(1 - alpha_t)
    return (beta_t**2/(2*sigma_t2*alpha_t*(1 - alpha_prod_t))) * ((true_eps - pred_eps)**2)

def lower_t_l2_loss(true_eps, pred_eps):
    loss = nn.MSELoss()
    return loss(true_eps, pred_eps)