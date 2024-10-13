import os
import argparse
import logging
import sys
sys.path.append("..")

import time
import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from model import DAIE_Model
from dataset import DAIE_Dataset, PadCollate
from train import DAIE_Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CLIP_PATH = "./openai/clip-vit-base-patch32"
DATA_PATH = {
                'train': './twitter/dataset_text/traindep.json',
                'val': './twitter/dataset_text/valdep.json',
                'test': './twitter/dataset_text/testdep.json',
            }
# image data
IMG_PATH = './twitter/dataset_image'

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='DAIE_dataset', type=str, help="The name of dataset.")
    parser.add_argument('--model_name', default='DAIE', type=str, help="The name of model.")
    parser.add_argument('--num_epochs', default=10, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=4, type=int, help="batch size")
    parser.add_argument('--clip_lr', default=5e-6, type=float, help="learning rate")
    parser.add_argument('--model_lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--linear_lr', default=5e-4, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--do_train', default=True , action='store_true')
    parser.add_argument('--only_test', default=False ,action='store_true')

    parser.add_argument('--mlp_output', default=300, type=int, help="dimension of MLP output")
    parser.add_argument('--gat_layer', default=3, type=int, help="The gat_layer of textual graph and visual graph")
    parser.add_argument('--dropout', default=0.2, type=float, help="all dropout")
    parser.add_argument('--beta', default=0.05, type=float, help="beta is the weight of loss_tlcl")
    parser.add_argument('--theta', default=0.5, type=float, help="theta is the weight of loss_glcl")
    parser.add_argument('--gamml', default=0.1, type=float, help="gamml is the weight of loss_imgs_pri")
    parser.add_argument('--lam', default=1, type=float, help="lam")
    parser.add_argument('--temperature', default=0.7, type=float, help="temperature parameter of contrastive learning")
    parser.add_argument('--imgpri', default=True, help="have TLCL_PRI Yes(True) or No(False)")
    parser.add_argument('--nsw', default=True, help="have the GLCL_NSW Yes(True) or No(False)")

    args = parser.parse_args()

    # path
    data_path, img_path, clip_path = DATA_PATH, IMG_PATH, CLIP_PATH

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        args.time_new = time.strftime('%Y%m%d%H%M%S', time.localtime())
        args.save_path = os.path.join(args.save_path, args.time_new)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    # logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    writer = None

    # Dataset
    train_dataset = DAIE_Dataset(data_path, img_path, clip_path, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, \
                                    collate_fn=PadCollate(clip_path=clip_path), pin_memory=True)
    dev_dataset = DAIE_Dataset(data_path, img_path, clip_path, mode='val')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, \
                                 collate_fn=PadCollate(clip_path=clip_path), pin_memory=True)
    test_dataset = DAIE_Dataset(data_path, img_path, clip_path, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, \
                                 collate_fn=PadCollate(clip_path=clip_path), pin_memory=True)

    # model
    model = DAIE_Model(args=args, clip_path=clip_path)
    trainer = DAIE_Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, args=args, logger=logger, writer=writer)
    
    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join(args.save_path, 'best_model.pth')
        # trainer.test()

    if args.only_test:
        # only do test
        args.load_path = os.path.join(args.load_path, 'best_model.pth')
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()
    

if __name__ == "__main__":
    main()