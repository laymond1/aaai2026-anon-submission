# This code is anonymized.

from typing import Tuple

import torch
from models.prompt_utils.vit import Attention, Block, VisionTransformer
from models.cps_prompt_utils.cps import CriticalPatchSampling


class CPSBlock(Block):
    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # Full Token Forward
        x_attn, attn_scores = self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt)
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        
        return x, attn_scores


class CPSAttention(Attention):
    def forward(
        self, x: torch.Tensor, register_hook: bool = False, prompt: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Full Token Forward
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # import ipdb; ipdb.set_trace()
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk,k), dim=2)
            v = torch.cat((pv,v), dim=2)

        attn_logit = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_logit.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
    
        if register_hook:
            v_norm = torch.linalg.norm(
                v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
            )  # value norm of size [B x T]
            significance_score = attn[:, :, 0].sum(
                dim=1
            )  # attention weights of CLS token of size [B x T]
            significance_score = significance_score * v_norm  # [B x T]
            significance_score = significance_score[:, 1:]  # [B x T-1]

            return x, significance_score
        else:
            return x, None


def make_cps_class(transformer_class):
    class CPSVisionTransformer(transformer_class):

        def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, feat=False, q_attn_scores=None) -> torch.Tensor:
            self.patchsampling = CriticalPatchSampling(reduction_ratio=self.reduction_ratio, sampling=self.sampling, token_shuffling=False, temperature=self.temperature)

            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
    
            x = x + self.pos_embed[:,:x.size(1),:]
            x = self.pos_drop(x)

            # Forward for prompt 
            if prompt is not None:
                if train:
                    # Sparse Patches Forward (Train for Prompt)
                    x = self.patchsampling(x, q_attn_scores)
                elif feat:
                    # Full Patches Forward (Train for Classifier)
                    pass
                else:
                    # Full Patches Forward (Inference)
                    pass
            # Forward for query
            else:
                pass

            prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)

            for i,blk in enumerate(self.blocks):

                if prompt is not None:
                    if train:
                        p_list, loss, x = prompt.forward(q, i, x, train=True)
                        prompt_loss = prompt_loss + loss
                    else:
                        p_list, _, x = prompt.forward(q, i, x, train=False)
                
                else:
                    p_list = None

                if prompt is not None:
                    x, attn_scores = blk(x, register_hook=(i == register_blk), prompt=p_list) # attn_scores is None
                else:
                    x, attn_scores = blk(x, register_hook=(i == 11), prompt=p_list) # attn_scores is only for the last block

            x = self.norm(x)

            if prompt is not None:
                prompt_loss /= len(prompt.e_layers)
            
            return x, prompt_loss, attn_scores

    return CPSVisionTransformer


def apply_patch(model: VisionTransformer):
    """
    Applies CPS-Prompt to this transformer.
    Args:
        model (VisionTransformer): The model to apply CPS-Prompt to.
    """
    CPSVisionTransformer = make_cps_class(model.__class__)

    model.__class__ = CPSVisionTransformer
    model.reduction_ratio = 0.0
    model.temperature = 1.0
    model.sampling = 'critical_score'
    model._cps_info = {
        "sampling": model.sampling,
        "token_shuffling": False,
        "temperature": model.temperature,
        "class_token": model.cls_token is not None,
        "distill_token": False
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._cps_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = CPSBlock
        elif isinstance(module, Attention):
            module.__class__ = CPSAttention
            module._cps_info = model._cps_info