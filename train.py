import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from model.stylenet import Stylenet
from fashion144k_loader import Fashion144kDataset
from utils import *

DATADIR = "/home/david/Programming/data/Fashion144k_stylenet_v1/"

parser = argparse.ArgumentParser(description='Stylenet Training')
parser.add_argument('--data-dir', type=str, default=DATADIR,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--num-workers', type=int, default=4,
                    help='dataloader workers')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-sample validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--preload', action='store_true',
                    help='preload images into memory')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
print("Using CUDA: {} on {} devices".format(use_cuda, ngpu))

ds = Fashion144kDataset(args.data_dir)
dl = data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                     shuffle=False, drop_last=True)

model = Stylenet(num_classes=ds.n_feats)

if use_cuda:
    model = nn.DataParallel(model).cuda() if ngpu > 1 else model.cuda()

model = model.cuda() if use_cuda else model

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=False)

losses = []
for epoch in range(args.epochs):
    model.train()
    losses_epoch = []
    for mb, tgts in dl:
        model.zero_grad()
        tgts = tgts.float()
        if use_cuda:
            mb, tgts = mb.cuda(), tgts.cuda()
        mb, tgts = Variable(mb), Variable(tgts)
        out = model(mb)
        loss = criterion(out, tgts)
        print("Loss: {0:.05f}".format(loss.data[0]))
        loss.backward()
        optimizer.step()
        losses_epoch += [loss.data[0]]
    losses.append(losses_epoch)

state = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_prec': None,
    'optimizer' : optimizer.state_dict(),
}

save_loss_csv(losses, "train_loss.csv")

save_checkpoint(state, False, "checkpts/stylenet_epoch_{}.pth".format(epoch))
