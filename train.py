import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models.fishnet import FishNet
from util import seed_everything
from dataset import GetCIFAR10
from trainer import Trainer

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1, help="gpu id")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--step_size", type=int, default=30, help="step size for lr_scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="multiplicative factor of learning rate decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--n_epochs", type=int, default=100, help="training epochs")
    
    parser.add_argument("--n_class", type=int, default=10, help="classs")
    parser.add_argument("--ratio", type=float, default=0.8, help="proportion of train data")
    parser.add_argument("--in_ch", type=int, default=64, help="input channels")
    parser.add_argument("--n_stage", type=int, default=3, help="the number of each stage(tail, body, head) layers")
    
    parser.add_argument("--out", default=True, help="Save model or not")
    parser.add_argument("--DATA_PATH", default='/home/yeji/FishNet/cifar10/cifar-10-batches-py/', help="data path")
    parser.add_argument("--MODEL_PATH", default='./outputs/', help="model path")
    parser.add_argument("--MODEL", default='base_FishNet', help="model name")
    parser.add_argument("--finetuning", default=False, help="Load pretrained model or not")    
    parser.add_argument("--PRETRAINED_MODEL", default='base_FishNet', help="pretrained model name")

    # set device and parameters
    config = parser.parse_args()

    return config


def main(config) :
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    # seed for Reproducibility
    seed_everything(config.seed)

    # load dataloader
    cifar10 = GetCIFAR10(config)
    train_loader, valid_loader = cifar10.get_loader()

    # set model
    if config.finetuning: 
        model = torch.load('{}{}.pt'.format(config.MODEL_PATH, config.PRETRAINED_MODEL)).to(device)
    else:
        model = FishNet(config.n_stage, config.in_ch, config.n_class, config.dropout).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), config.lr, momentum=config.momentum, 
                          weight_decay=config.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, 
                                          gamma=config.gamma)

    trainer = Trainer(model, optimizer, scheduler, loss_function, config)

    trainer.run(train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)