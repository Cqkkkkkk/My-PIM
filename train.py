import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from eval import  cal_train_metrics
from config import cfg


def train_epoch(epoch, model, scaler, amp_context, optimizer, scheduler, train_loader):

    optimizer.zero_grad()
    total_batchs = len(train_loader)  # just for log
    show_progress = [x / 10 for x in range(11)]  # just for log
    progress_i = 0

    for batch_id, (_, datas, labels) in enumerate(train_loader):
        model.train()
        # LR Scheduler
        iterations = epoch * len(train_loader) + batch_id
        scheduler.step(iterations, optimizer)

        batch_size = labels.size(0)

        # Model forward
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
                        logit = outs[name].view(-1, cfg.datasets.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit, labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += cfg.model.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                elif "drop_" in name:
                    if not cfg.model.use_selection:
                        raise ValueError("Selector not use here.")

                    if cfg.model.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, cfg.datasets.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, cfg.datasets.num_classes]) - 1
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
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
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
            msg['info/lr'] = scheduler.get_lr(optimizer)
            cal_train_metrics(msg, outs, labels, batch_size)
            wandb.log(msg)

        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(
                ".."+str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1
