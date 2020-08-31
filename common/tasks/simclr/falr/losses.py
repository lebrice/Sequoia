import torch
import torch.nn.functional as F


class SimCLRLoss(torch.nn.Module):
  def __init__(self, dim: int, use_bilinear: bool = False):
    super().__init__()
    self.W = torch.nn.Linear(dim, dim, use_bilinear)
    self.use_bilinear = use_bilinear

  def forward(self, z: torch.Tensor, xent_temp: float):
    z = F.normalize(z, p=2, dim=1)
    z2 = self.W(z) if self.use_bilinear else z
    cos_sim = torch.einsum('id,jd->ij', z, z2) / xent_temp
    cos_sim.fill_diagonal_(0) # Mask self-similarity
    targets = torch.zeros(len(z), dtype=torch.long).cuda()
    targets[0::2] = torch.arange(1,z.shape[0],2)
    targets[1::2] = torch.arange(0,z.shape[0]-1,2)
    return F.cross_entropy(cos_sim, targets, reduction='mean')


def class_sim_loss(z: torch.Tensor, y: torch.Tensor, xent_temp: float):
  z = F.normalize(z)
  cos_sim = torch.einsum('id,jd->ij', z, z) / xent_temp
  mask = (y.unsqueeze(-1) == y).float()
  return F.binary_cross_entropy_with_logits(cos_sim, mask)
