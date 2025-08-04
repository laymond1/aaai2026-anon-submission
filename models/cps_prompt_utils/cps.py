# This code is anonymized.

import torch


class CriticalPatchSampling(torch.nn.Module):
    """
    This is a Critical Patch Sampling Module.
    """
    def __init__(self, reduction_ratio=0.5, sampling="uniform", token_shuffling=False, temperature=1.0):
        super().__init__()
        assert 0 <= reduction_ratio < 1, "The reduction_ratio must be in [0,1)"
        
        self.reduction_ratio = reduction_ratio
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.temperature = temperature

    def forward(self, x, attn_scores=None, force_drop=True):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        # if not self.training and not force_drop: return x        
        if not force_drop: return x        
        if self.reduction_ratio == 0: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x, attn_scores)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x, attn_scores=None):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        elif self.sampling == "critical_score":
            return self.critical_score_mask(x, attn_scores)
        else:
            return NotImplementedError(f"CPS does ot support {self.sampling} sampling")
    
    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        keep = int(_L * (1-self.reduction_ratio))
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
    
    def critical_score_mask(self, x, attn_scores):
        """
        Returns an id-mask using attention scores
        """
        N, L, D = x.shape
        _L = L - 1
        keep = int(_L * (1-self.reduction_ratio))

        # Apply temperature scaling
        attn_scores = attn_scores / self.temperature
        attn_scores = torch.softmax(attn_scores, dim=1)

        # Sample 'keep' tokens from 196 using attention probs
        patch_mask = torch.multinomial(attn_scores, num_samples=keep, replacement=False)  # (B, keep)
        patch_mask = patch_mask + 1
        
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]

        return patch_mask