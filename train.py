import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from model.stylenet import *
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

model_fe = FENet(num_features=128)
model_cl = ClassificationNet(num_in_features=128, num_out_classes=ds.n_feats)

if use_cuda:
    model_fe = nn.DataParallel(model_fe).cuda() if ngpu > 1 else model_fe.cuda()
    model_cl = nn.DataParallel(model_cl).cuda() if ngpu > 1 else model_cl.cuda()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD([
                {'params': model_fe.parameters()},
                {'params': model_cl.parameters()}
            ], lr=0.0001, momentum=0.9, nesterov=False)

losses = []
for epoch in range(args.epochs):
    model_fe.train()
    model_cl.train()
    losses_epoch = []
    for mb, tgts in dl:
        model_fe.zero_grad()
        model_cl.zero_grad()
        tgts = tgts.float()
        if use_cuda:
            mb, tgts = mb.cuda(), tgts.cuda()
        mb, tgts = Variable(mb), Variable(tgts)
        features = model_fe(mb)
        out = model_cl(features)
        loss = criterion(out, tgts)
        print("Loss: {0:.05f}".format(loss.data[0]))
        loss.backward()
        optimizer.step()
        losses_epoch += [loss.data[0]]
    losses.append(losses_epoch)

state = {
    'epoch': epoch + 1,
    'state_dict_fe': model_fe.state_dict(),
    'state_dict_cl': model_cl.state_dict(),
    'best_prec': None,
    'optimizer' : optimizer.state_dict(),
}

save_loss_csv(losses, "train_loss.csv")

save_checkpoint(state, False, "checkpts/stylenet_epoch_{}.pth".format(epoch))
