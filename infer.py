import torch
import warnings
import timm
from data.dataset import build_loader
from utils.logger import timeLogger
from utils.record import build_record_folder
from models.pim_module import PluginMoodel
from eval import eval_and_cm

from cmd_args import parse_args
from config import cfg


warnings.simplefilter("ignore")


def set_environment(tlogger):
    print("Setting Environment...")

    cfg.train.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ### = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    _, val_loader = build_loader()

    print("[Only Evaluation]")

    tlogger.print()

    ### = = = =  Model = = = =
    tlogger.print("Building Model....")

    backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
    model = PluginMoodel(backbone=backbone,
                        return_nodes=None,
                        img_size=cfg.datasets.data_size,
                        fpn_size=cfg.model.fpn_size,
                        proj_type='Linear',
                        upsample_type='Conv',
                        use_selection=cfg.model.use_selection,
                        num_classes=cfg.datasets.num_classes,
                        num_selects=dict(zip(cfg.model.num_selects_layer_names, cfg.model.num_selects)),
                        use_combiner=cfg.model.use_combiner)

    checkpoint = torch.load(cfg.model.pretrained, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])


    model.to(cfg.train.device)
    tlogger.print()

    return val_loader, model


def main_test(tlogger):
    val_loader, model = set_environment(tlogger)
    eval_and_cm(model, val_loader, tlogger)


if __name__ == "__main__":
    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)

    build_record_folder()
    tlogger.print()

    main_test(tlogger)