from yacs.config import CfgNode as CN


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name
    :return: configuration use by the experiment.
    '''
    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    cfg.project = CN()
    cfg.project.name = 'CUB200'
    cfg.project.exp_name = 'T300'

    # ------------------------------------------------------------------------ #
    # Wandb options
    # ------------------------------------------------------------------------ #
    cfg.wandb = CN()
    cfg.wandb.use = False
    cfg.wandb.entity = 'chenqin'

    # ------------------------------------------------------------------------ #
    # Datasets 
    # ------------------------------------------------------------------------ #
    cfg.datasets = CN()
    cfg.datasets.train_root = '../datasets/CUB/sortedImages/train/'
    cfg.datasets.val_root = '../datasets/CUB/sortedImages/test/'
    cfg.datasets.data_size = 384
    cfg.datasets.num_workers = 2
    cfg.datasets.batch_size = 8
    cfg.datasets.num_classes = 200
    
    # ------------------------------------------------------------------------ #
    # Backbone Model 
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    cfg.model.name = 'swin-t'
    cfg.model.pretrained = None
    cfg.model.use_amp = True
    cfg.model.use_fpn = True
    cfg.model.fpn_size = 1536
    cfg.model.use_selection = True
    # For layer 1, 2, 3, 4
    cfg.model.num_selects_layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    cfg.model.num_selects = [2048, 512, 128, 32] 
    cfg.model.use_combiner = True
    cfg.model.lambda_b = 0.5
    cfg.model.lambda_s = 0.0
    cfg.model.lambda_n = 5.0
    cfg.model.lambda_c = 1.0

    # ------------------------------------------------------------------------ #
    # Optimization options 
    # ------------------------------------------------------------------------ #
    cfg.optim = CN()
    cfg.optim.optimizer = 'SGD'
    cfg.optim.max_lr = 5e-4
    cfg.optim.wd = 5e-4
    cfg.optim.epochs = 50
    cfg.optim.warmup_batches = 800
    
    
    # ------------------------------------------------------------------------ #
    # Training options 
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    cfg.train.device = 'cuda'
    cfg.train.update_freq = 2
    cfg.train.eval_freq = 10
    cfg.train.save_dir = ''
    # ------------------------------------------------------------------------ #
    # log options 
    # ------------------------------------------------------------------------ #
    cfg.log = CN()
    cfg.log.log_freq = 100
    

# Global config object
cfg = CN()

set_cfg(cfg)