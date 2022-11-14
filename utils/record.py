import os
from config import cfg


def build_record_folder():

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")

    cfg.train.save_dir = "./records/" + \
        cfg.project.name + "/" + cfg.project.exp_name + "/"
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    os.makedirs(cfg.train.save_dir + "backup/", exist_ok=True)
