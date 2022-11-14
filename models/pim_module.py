import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import pdb

from models.combiner import GCNCombiner
from models.selector import WeaklySelector
from models.fpn import FPN


class PluginMoodel(nn.Module):
    """
        * backbone: 
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes: 
            tuple or list contain features map size of each layers. 
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn: 
            boolean, use features pyramid network or not
        * fpn_size: 
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }

        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which 
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer 
        could be different).
    """
    def __init__(self,
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 num_selects: dict,
                 use_combiner: bool,
                 comb_proj_size: Union[int, None],
                 positive_adj : bool
                 ):
       
        super(PluginMoodel, self).__init__()

        # = = = = = Backbone = = = = =
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(
                backbone, return_nodes=return_nodes)
        else:
            self.backbone = backbone

        # get hidden feartues size
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = self.backbone(rand_in)

        # just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size.size(-1)
                elif len(fs_size) == 4:
                    out_size = fs_size.size(1)
                else:
                    raise ValueError(
                        "The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

        # = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        # = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            # if not using fpn, build classifier in weakly selector
            w_fpn_size = self.fpn_size if self.use_fpn else None
            self.selector = WeaklySelector(
                outs, num_classes, num_selects, w_fpn_size)

        # = = = = = Combiner = = = = =

        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = outs, comb_proj_size  # redundant, fix in future
            total_num_selects = sum([num_selects[name]
                                    for name in num_selects])  # sum
            self.combiner = GCNCombiner(
                total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size, positive_adj)

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1)
            )
            self.add_module("fpn_classifier_"+name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            # predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_"+name)(logit)
            logits[name] = logits[name].transpose(
                1, 2).contiguous()  # transpose

    def forward(self, x: torch.Tensor):

        logits = {}

        x = self.forward_backbone(x)

        if self.use_fpn:
            x = self.fpn(x)
            self.fpn_predict(x, logits)

        if self.use_selection:
            selects = self.selector(x, logits)

        if self.use_combiner:
            comb_outs = self.combiner(selects)
            logits['comb_outs'] = comb_outs
            return logits

        if self.use_selection or self.fpn:
            return logits

        # original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits['ori_out'] = logits

        return logits
