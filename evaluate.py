import os
import argparse

import torch
import torch.nn as nn

from util import seed_everything
from dataset import GetCIFAR10
from trainer import Trainer

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1, help="gpu id")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
        
    parser.add_argument("--DATA_PATH", default='/home/yeji/FishNet/cifar10/cifar-10-batches-py/', help="Data path")
    parser.add_argument("--MODEL_PATH", default='./outputs/', help="Model path")
    parser.add_argument("--MODEL", default='base_FishNet', help="Model name")

    # set device and parameters
    config = parser.parse_args()

    return config


def main(config) :
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)   

    # seed for Reproducibility
    seed_everything(config.seed)

    # load dataloader
    cifar10 = GetCIFAR10(config)
    test_loader = cifar10.get_test()

    # set model
    input_path = os.path.join(config.MODEL_PATH, config.MODEL)
    model = torch.load(input_path).to(device)
    
    loss_function = nn.CrossEntropyLoss()

    trainer = Trainer(model, None, None, loss_function, config)

    acc, _ = trainer.validate(test_loader)
    print("Test ACC = %.4f" % (acc))

if __name__ == '__main__':
    config = define_argparser()
    main(config)