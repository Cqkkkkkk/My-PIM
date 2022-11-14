import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics

from cmd_args import parse_args
from config import cfg

warnings.simplefilter("ignore")


def eval_freq_schedule(epoch: int):
    if epoch >= cfg.optim.epochs * 0.95:
        cfg.train.eval_freq = 1
    elif epoch >= cfg.optim.epochs * 0.9:
        cfg.train.eval_freq = 1
    elif epoch >= cfg.optim.epochs * 0.8:
        cfg.train.eval_freq = 2


def set_environment(tlogger):
    print("Setting Environment...")
     ### = = = =  Dataset and Data Loader = = = =  
    tlogger.print("Building Dataloader....")
    
    train_loader, val_loader = build_loader(args)
    
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()



if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    
    tmp = dict(zip(cfg.model.num_selects_layer_names, cfg.model.num_selects_val))
    print(tmp)

