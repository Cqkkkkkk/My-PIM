import torch
import timm
import contextlib
import wandb
import warnings


from data.dataset import build_loader
from utils.logger import timeLogger
from utils.record import build_record_folder
from utils.scheduler import CosineDecayLRScheduler, eval_freq_schedule
from eval import evaluate, eval_and_save
from train import train_epoch
from cmd_args import parse_args
from config import cfg
from models.pim_module import PluginMoodel


import pdb

warnings.simplefilter("ignore")


def set_environment(tlogger):

    print("Setting Environment...")
    cfg.train.device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------------ #
    # Dataset and Data Loader
    # ------------------------------------------------------------------------ #

    tlogger.print("Building Dataloader....")
    train_loader, val_loader = build_loader()

    tlogger.print()

    # ------------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------------ #

    tlogger.print("Building Model....")

    if cfg.model.name == 'swin-t':

        backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)

        model = PluginMoodel(backbone=backbone,
                             return_nodes=None,
                             img_size=cfg.datasets.data_size,
                             use_fpn=cfg.model.use_fpn,
                             fpn_size=cfg.model.fpn_size,
                             proj_type='Linear',
                             upsample_type='Conv',
                             use_selection=cfg.model.use_selection,
                             num_classes=cfg.datasets.num_classes,
                             num_selects=dict(zip(cfg.model.num_selects_layer_names, cfg.model.num_selects)),
                             use_combiner=cfg.model.use_combiner)

    start_epoch = 0
    if cfg.model.pretrained is not None:
        ckpt = torch.load(cfg.model.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        

    model.to(cfg.train.device)
    tlogger.print()


    # ------------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------------ #

    tlogger.print("Building Optimizer....")
    if cfg.optim.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.optim.max_lr,
                                    nesterov=True,
                                    momentum=0.9,
                                    weight_decay=cfg.optim.wd)
    elif cfg.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.max_lr, weight_decay=cfg.optim.wd)
    elif cfg.optim.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.max_lr, weight_decay=cfg.optim.wd)

    if cfg.model.pretrained is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = CosineDecayLRScheduler(len(train_loader))

    scaler = torch.cuda.amp.GradScaler()
    amp_context = torch.cuda.amp.autocast

    return train_loader, val_loader, model, optimizer, scheduler, scaler, amp_context, start_epoch


# Train the model with options in config, saving the last and best model
def main(tlogger):

    train_loader, val_loader, model, optimizer, \
        scheduler, scaler, amp_context, start_epoch = set_environment(tlogger)

    best_acc = 0.0
    best_eval_name = "null"

    if cfg.wandb.use:
        wandb.init(entity=cfg.wandb.entity,
                   project=cfg.project.name,
                   name=cfg.project.exp_name,
                   config=cfg)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    # Train
    for epoch in range(start_epoch, cfg.optim.epochs):

        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train_epoch(epoch, model, scaler, amp_context, optimizer, scheduler, train_loader)
            tlogger.print()
        else:
            eval_and_save(model, val_loader)
            break

        eval_freq_schedule(epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        # torch.save(checkpoint, cfg.train.save_dir + "backup/last.pt")

        # Evaluation
        if epoch == 0 or (epoch + 1) % cfg.train.eval_freq == 0:
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(model, val_loader)
                tlogger.print("....BEST_ACC: {}% ({}%)".format(
                    max(acc, best_acc), acc))
                tlogger.print()

            if cfg.wandb.use:
                wandb.log(accs)

            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, cfg.train.save_dir + "backup/best.pt")
            if cfg.wandb.use:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1


if __name__ == "__main__":

    tlogger = timeLogger()
    tlogger.print("Reading Config...")

    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    print(cfg)
    build_record_folder()
    tlogger.print()

    main(tlogger)
