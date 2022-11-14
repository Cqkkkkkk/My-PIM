import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.record import build_record_folder
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics, eval_and_save

from cmd_args import parse_args
from config import cfg


import pdb

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

    cfg.train.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # = = = =  Dataset and Data Loader = = = =
    tlogger.print("Building Dataloader....")

    train_loader, val_loader = build_loader()

    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(
            len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(
            len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    # = = = =  Model = = = =
    tlogger.print("Building Model....")
    model = MODEL_GETTER[cfg.model.name](
        use_fpn=cfg.model.use_fpn,
        fpn_size=cfg.model.fpn_size,
        use_selection=cfg.model.use_selection,
        num_classes=cfg.datasets.num_classes,
        num_selects=dict(
            zip(cfg.model.num_selects_layer_names, cfg.model.num_selects)),
        use_combiner=cfg.model.use_combiner,
    )  # about return_nodes, we use our default setting

    if cfg.model.pretrained is not None:
        checkpoint = torch.load(cfg.model.pretrained,
                                map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    model.to(cfg.train.device)
    tlogger.print()

    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None

    # = = = =  Optimizer = = = =
    tlogger.print("Building Optimizer....")
    if cfg.optim.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=cfg.optim.max_lr, nesterov=True, momentum=0.9, weight_decay=cfg.optim.wd)
    elif cfg.optim.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.max_lr)

    if cfg.model.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    schedule = cosine_decay(len(train_loader))

    if cfg.model.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(epoch, model, scaler, amp_context, optimizer, schedule, train_loader):

    optimizer.zero_grad()
    total_batchs = len(train_loader)  # just for log
    show_progress = [x/10 for x in range(11)]  # just for log
    progress_i = 0
    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        """ = = = = adjust learning rate = = = = """
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        datas, labels = datas.to(cfg.train.device), labels.to(cfg.train.device)

        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            outs = model(datas)

            loss = 0.
            for name in outs:

                # pdb.set_trace()
                if "select_" in name:
                    if not cfg.model.use_selection:
                        raise ValueError("Selector not use here.")
                    if cfg.model.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1,
                                                cfg.datasets.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += cfg.model.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not cfg.model.use_selection:
                        raise ValueError("Selector not use here.")

                    if cfg.model.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1,
                                                cfg.datasets.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros(
                            [batch_size * S, cfg.datasets.num_classes]) - 1
                        labels_0 = labels_0.to(cfg.train.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += cfg.model.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                elif "layer" in name:
                    if not cfg.model.use_fpn:
                        raise ValueError("FPN not use here.")
                    if cfg.model.lambda_b != 0:
                        # here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(
                            outs[name].mean(1), labels)
                        loss += cfg.model.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                elif "comb_outs" in name:
                    if not cfg.model.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if cfg.model.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += cfg.model.lambda_c * loss_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            loss /= cfg.train.update_freq

        """ = = = = calculate gradient = = = = """
        if cfg.model.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        if (batch_id + 1) % cfg.train.update_freq == 0:
            if cfg.model.use_amp:
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                optimizer.step()
            optimizer.zero_grad()

        """ log (MISC) """
        if cfg.wandb.use and ((batch_id + 1) % cfg.log.log_freq == 0):
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            cal_train_metrics(msg, outs, labels, batch_size)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(
                ".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1


# Train the model with options in config, saving the last and best model
def main(tlogger):

    train_loader, val_loader, model, optimizer, \
        schedule, scaler, amp_context, start_epoch = set_environment(tlogger)

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
            train(epoch, model, scaler, amp_context,
                  optimizer, schedule, train_loader)
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
        torch.save(checkpoint, cfg.train.save_dir + "backup/last.pt")

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

    build_record_folder()
    tlogger.print()

    main(tlogger)
