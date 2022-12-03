import torch
from typing import Union
import timm
from models.pim_module import PluginMoodel

"""
[Default Return]
Set return_nodes to None, you can use default return type, all of the model in this script 
return four layers features.

[Model Configuration]
if you are not using FPN module but using Selector and Combiner, you need to give Combiner a 
projection  dimension ('proj_size' of GCNCombiner in pim_module.py), because graph convolution
layer need the input features dimension be the same.

[Combiner]
You must use selector so you can use combiner.

[About Costom Model]
This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
could cause error, so we set return_nodes to None and change swin-transformer model script to
return features directly.
Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
model also fail at create_feature_extractor or get_graph_node_names step.
"""


def load_model_weights(model, model_path):
    # reference https://github.com/TACJu/TransFG
    # thanks a lot.
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(
                    key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model


def build_resnet50(pretrained: str = "./resnet50_miil_21k.pth",
                   return_nodes: Union[dict, None] = None,
                   num_selects: Union[dict, None] = None,
                   img_size: int = 448,
                   use_fpn: bool = True,
                   fpn_size: int = 512,
                   proj_type: str = "Conv",
                   upsample_type: str = "Bilinear",
                   use_selection: bool = True,
                   num_classes: int = 200,
                   use_combiner: bool = True,
                   comb_proj_size: Union[int, None] = None,
                   ):

    

    if return_nodes is None:
        return_nodes = {
            'layer1.2.act3': 'layer1',
            'layer2.3.act3': 'layer2',
            'layer3.5.act3': 'layer3',
            'layer4.2.act3': 'layer4',
        }
    if num_selects is None:
        num_selects = {
            'layer1': 32,
            'layer2': 32,
            'layer3': 32,
            'layer4': 32
        }

    backbone = timm.create_model('resnet50', pretrained=False, num_classes=11221)
    # original pretrained path "./models/resnet50_miil_21k.pth"
    if pretrained != "":
        backbone = load_model_weights(backbone, pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))

    return PluginMoodel(backbone=backbone,
                        return_nodes=return_nodes,
                        img_size=img_size,
                        use_fpn=use_fpn,
                        fpn_size=fpn_size,
                        proj_type=proj_type,
                        upsample_type=upsample_type,
                        use_selection=use_selection,
                        num_classes=num_classes,
                        num_selects=num_selects,
                        use_combiner=num_selects,
                        comb_proj_size=comb_proj_size,
                        
                        )




def build_swintransformer(pretrained: bool = True,
                          num_selects: Union[dict, None] = None,
                          img_size: int = 384,
                          use_fpn: bool = True,
                          fpn_size: int = 512,
                          proj_type: str = "Linear",
                          upsample_type: str = "Conv",
                          use_selection: bool = True,
                          num_classes: int = 200,
                          use_combiner: bool = True,
                          comb_proj_size: Union[int, None] = None,
                          ):
    """
    This function is to building swin transformer. timm swin-transformer + torch.fx.proxy.Proxy 
    could cause error, so we set return_nodes to None and change swin-transformer model script to
    return features directly.
    Please check 'timm/models/swin_transformer.py' line 541 to see how to change model if your costom
    model also fail at create_feature_extractor or get_graph_node_names step.
    """
    
    if num_selects is None:
        num_selects = {
            'layer1': 32,
            'layer2': 32,
            'layer3': 32,
            'layer4': 32
        }

    backbone = timm.create_model(
        'swin_large_patch4_window12_384_in22k', pretrained=pretrained)

    # print(backbone)
    # print(get_graph_node_names(backbone))
    backbone.train()

    print("Building...")
    return PluginMoodel(backbone=backbone,
                        return_nodes=None,
                        img_size=img_size,
                        use_fpn=use_fpn,
                        fpn_size=fpn_size,
                        proj_type=proj_type,
                        upsample_type=upsample_type,
                        use_selection=use_selection,
                        num_classes=num_classes,
                        num_selects=num_selects,
                        use_combiner=num_selects,
                        comb_proj_size=comb_proj_size,
                        )


MODEL_GETTER = {
    "resnet50": build_resnet50,
    "swin-t": build_swintransformer,
}


if __name__ == "__main__":
    pass