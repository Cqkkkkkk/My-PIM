import torch
import torch.nn as nn
from typing import Union
import pdb


class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        self.fpn_size = fpn_size
        # build classifier
        if self.fpn_size is None:
            self.num_classes = num_classes
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                m = nn.Linear(in_size, num_classes)
                self.add_module("classifier_l_"+name, m)

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

    def forward(self, x, logits=None):
        """
        x : 
            dictionary contain the features maps which 
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        if self.fpn_size is None:
            logits = {}
        selections = {}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])

            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1 = []
            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                selections[name].append(sf)  # [num_selected, C]
                preds_1.append(logits[name][bi][ranks[:num_select]])
                preds_0.append(logits[name][bi][ranks[num_select:]])

            selections[name] = torch.stack(selections[name])
            preds_1 = torch.stack(preds_1)
            preds_0 = torch.stack(preds_0)

            logits["select_"+name] = preds_1
            logits["drop_"+name] = preds_0

        return selections
