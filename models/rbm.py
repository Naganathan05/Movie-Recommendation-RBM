import torch
from torch import nn

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, learning_rate=0.01):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        self.a = nn.Parameter(torch.zeros(num_hidden))  
        self.b = nn.Parameter(torch.zeros(num_visible))  
        self.learning_rate = learning_rate
    
    def sample_h(self, x):
        activation = torch.mm(x, self.W) + self.a
        prob_h = torch.sigmoid(activation)
        sample_h = torch.bernoulli(prob_h)
        return prob_h, sample_h
    
    def sample_v(self, h):
        activation = torch.mm(h, self.W.t()) + self.b
        prob_v = torch.sigmoid(activation)
        sample_v = torch.bernoulli(prob_v)
        return prob_v, sample_v
    
    def contrastive_divergence(self, v0, k=1):
        ph0, h0 = self.sample_h(v0)
        
        vk = v0.clone()
        for _ in range(k):
            _, hk = self.sample_h(vk)
            _, vk = self.sample_v(hk)
            mask = (v0 < 0).float()
            vk = vk * (1 - mask) + v0 * mask
        
        phk, _ = self.sample_h(vk)
        
        w_pos_grad = torch.mm(v0.t(), ph0)
        w_neg_grad = torch.mm(vk.t(), phk)
        
        self.W.data += self.learning_rate * (w_pos_grad - w_neg_grad)
        self.b.data += self.learning_rate * torch.sum(v0 - vk, dim=0)
        self.a.data += self.learning_rate * torch.sum(ph0 - phk, dim=0)
        
        mask = (v0 >= 0).float()
        valid_count = torch.sum(mask)
        if valid_count > 0:
            loss = torch.sum(mask * (v0 - vk)**2) / valid_count
        else:
            loss = torch.tensor(0.0)
        
        return loss
    
    def forward(self, v):
        h, _ = self.sample_h(v)
        v_recon, _ = self.sample_v(h)
        return v_recon