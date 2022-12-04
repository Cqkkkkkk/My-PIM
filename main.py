import torch
import timm
import wandb
import warnings


from data.dataset import build_loader
from utils.logger import timeLogger
from utils.record import build_record_folder
from utils.scheduler import CosineDecayLRScheduler, eval_freq_schedule
from eval import evaluate
from train import train_epoch
from cmd_args import parse_args
from config import cfg
from models.pim_module import PluginMoodel


import pdb

warnings.simplefilter("ignore")


class ModelTrainer:
    def __init__(self, tlogger) -> None:
        self.tlogger = tlogger
        print("Setting Environment...")
        cfg.train.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # ------------------------------------------------------------------------ #
        # Dataset and Data Loader
        # ------------------------------------------------------------------------ #
        tlogger.print("Building Dataloader....")
        self.train_loader, self.val_loader = build_loader()   

        # ------------------------------------------------------------------------ #
        # Model
        # ------------------------------------------------------------------------ #
        self.tlogger.print("Building Model....")

        if cfg.model.name == 'swin-t':
            backbone = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
            self.model = PluginMoodel(backbone=backbone,
                                return_nodes=None,
                                img_size=cfg.datasets.data_size,
                                fpn_size=cfg.model.fpn_size,
                                proj_type='Linear',
                                upsample_type='Conv',
                                use_selection=cfg.model.use_selection,
                                num_classes=cfg.datasets.num_classes,
                                num_selects=dict(zip(cfg.model.num_selects_layer_names, cfg.model.num_selects)),
                                use_combiner=cfg.model.use_combiner).to(cfg.train.device)
        self.start_epoch = 0
        if cfg.model.pretrained is not None:
            ckpt = torch.load(cfg.model.pretrained, map_location=torch.device('cpu'))
            self.model.load_state_dict(ckpt['model'])
            self.start_epoch = ckpt['epoch']

        # ------------------------------------------------------------------------ #
        # Optimizer
        # ------------------------------------------------------------------------ #

        self.tlogger.print("Building Optimizer....")
        if cfg.optim.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=cfg.optim.max_lr,
                                        nesterov=True,
                                        momentum=0.9,
                                        weight_decay=cfg.optim.wd)
        elif cfg.optim.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.optim.max_lr, weight_decay=cfg.optim.wd)
        elif cfg.optim.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.optim.max_lr, weight_decay=cfg.optim.wd)

        if cfg.model.pretrained is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])

        self.scheduler = CosineDecayLRScheduler(len(self.train_loader))

        self.scaler = torch.cuda.amp.GradScaler()
        self.amp_context = torch.cuda.amp.autocast


    def train_model(self):
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
        for epoch in range(self.start_epoch, cfg.optim.epochs):

           
            tlogger.print("Start Training {} Epoch".format(epoch+1))
            train_epoch(epoch, self.model, self.scaler, self.amp_context, self.optimizer, self.scheduler, self.train_loader)
            self.tlogger.print()
        
            eval_freq_schedule(epoch)

            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            checkpoint = {
                "model": model_to_save.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }
            # torch.save(checkpoint, cfg.train.save_dir + "backup/last.pt")

            # Evaluation
            if epoch == 0 or (epoch + 1) % cfg.train.eval_freq == 0:
            
                self.tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                acc, eval_name, accs = evaluate(self.model, self.val_loader)
            
                if cfg.wandb.use:
                    wandb.log(accs)

                if acc > best_acc:
                    best_acc = acc
                    best_eval_name = eval_name
                    torch.save(checkpoint, cfg.train.save_dir + "backup/best.pt")

                self.tlogger.print("....BEST_ACC: {}% ({}%)\n".format(best_acc, acc))

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

    trainer = ModelTrainer(tlogger)
    trainer.train_model()
