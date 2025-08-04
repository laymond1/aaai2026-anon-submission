"""
CPS-Prompt:Critical Patch-Aware Sparse Prompting with Decoupled Trianing for Continual Learning on the Edge.

Note:
    CPS-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.
    The backbone is a ViT-B/16 pretrained on Imagenet 21k and finetuned on ImageNet 1k.
"""

import torch
import torch.nn.functional as F

from datasets import get_dataset
from utils.args import ArgumentParser

from models.utils.continual_model import ContinualModel
from models.cps_prompt_utils.model import PromptModel
from utils.schedulers import CosineSchedule
from utils import binary_to_boolean_type


class CPSPrompt(ContinualModel):
    """CPS-Prompt:Critical Patch-Aware Sparse Prompting with Decoupled Trianing for Continual Learning on the Edge."""
    NAME = 'cps-prompt'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # Parameters
        parser.add_argument('--vit_type', type=str, default='tiny', choices=['tiny', 'small', 'base'], help='ViT type')
        parser.add_argument('--e_prompt_layer_idx', type=int, default=[0, 1, 2, 3, 4], nargs="+", help='the layer index of the E-Prompt')
        parser.add_argument('--e_prompt_pool_size', type=int, default=100, help='pool size')
        parser.add_argument('--e_prompt_length', type=int, default=8, help='prompt length')
        parser.add_argument('--ortho_mu', type=float, default=0.0, help='orthogonal penalty weight') # but it's set to 0.0 becuase of (#issue12)[https://github.com/GT-RIPL/CODA-Prompt/issues/12]
        parser.add_argument('--pull_constraint_coeff', type=float, default=1.0, help='Coefficient(mu) for the pull constraint term, \
                            controlling the weight of the prompt loss in the total loss calculation')

        # Critical Patch Sampling
        parser.add_argument('--reduction_ratio', type=float, default=0.5, help='the ratio of patches to reduce')
        parser.add_argument('--sampling', type=str, default='critical_score', choices=['uniform', 'critical_score'], help='sampling method for patch merging')
        parser.add_argument('--temperature', type=float, default=1.0, help='temperature for the attention scaling')
        # Decoupled Prompt and Classifier Training
        parser.add_argument('--phase_ratio', type=float, default=0.8, help='the ratio of the epochs to start training the classifier')

        # ETC
        parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
        parser.add_argument('--use_scheduler', type=binary_to_boolean_type, default=True, help='Use scheduler')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        del backbone
        print("-" * 20)
        print(f"WARNING: CODA-Prompt USES A CUSTOM BACKBONE: `vit_base_patch16_224`.")
        print("Pretrained on Imagenet 21k and finetuned on ImageNet 1k.")
        print("-" * 20)

        tmp_dataset = get_dataset(args) if dataset is None else dataset
        num_classes = tmp_dataset.N_CLASSES
        args.n_tasks = tmp_dataset.N_TASKS
        backbone = PromptModel(args, 
                               num_classes=num_classes,
                               pretrained=True, prompt_flag='coda',
                               prompt_param=[args.e_prompt_pool_size, args.e_prompt_length, args.ortho_mu])

        super().__init__(backbone, loss, args, transform, dataset=dataset)
    
    def begin_task(self, dataset):
        if self.current_task > 0:
            self.net.prompt.process_task_count()
        if hasattr(self, 'opt'):
            self.opt.zero_grad(set_to_none=True)
            del self.opt
        self.opt = self.get_optimizer()
        if self.args.use_scheduler:
            self.scheduler = CosineSchedule(self.opt, K=self.args.n_epochs)

    def begin_epoch(self, epoch, dataset):
        self.count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if epoch < int(self.args.n_epochs * self.args.phase_ratio):
            logits, loss_prompt = self.net(inputs, train=True)
        else:
            with torch.no_grad():
                feats = self.net(inputs, feat=True, train=False).detach()
            logits = self.net(feats, last=True)
            loss_prompt = None
        # here is the trick to mask out classes of non-current tasks
        logits[:, :self.n_past_classes] = -float('inf')

        loss = self.loss(logits[:, :self.n_seen_classes], labels)
        if self.args.pull_constraint_coeff > 0.0 and loss_prompt is not None:
            loss = loss + self.args.pull_constraint_coeff * loss_prompt.mean() # the mean is needed for data-parallel (concatenates instead of averaging)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.args.clip_grad)
        self.opt.step()

        # Calculate accuracy
        preds = torch.argmax(logits[:, :self.n_seen_classes], dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total

        # Update running loss와 accuracy
        self.count += 1
        self.running_loss += loss.item()
        self.running_accuracy += accuracy

        return loss.item()
    
    def get_parameters(self):
        return [p for n, p in self.net.named_parameters() if 'prompt' in n or 'head' in n]
    
    def forward(self, x):
        return self.net(x)[:, :self.n_seen_classes]