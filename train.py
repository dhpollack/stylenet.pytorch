import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from model.stylenet import Stylenet
from fashion144k_loader import Fashion144kDataset

DATADIR = "/home/david/Programming/data/Fashion144k_stylenet_v1/"

parser = argparse.ArgumentParser(description='Stylenet Training')
parser.add_argument('--data-dir', type=str, default=DATADIR,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=10,
                    help='batch size')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print("Using CUDA: {}".format(use_cuda))

ds = Fashion144kDataset(args.data_dir)
dl = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

model = Stylenet(num_classes=ds.n_feats)
model = model.cuda() if use_cuda else model

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=False)

for epoch in range(args.epochs):
    model.train()
    for mb, tgts in dl:
        model.zero_grad()
        tgts = tgts.float()
        if use_cuda:
            mb, tgts = mb.cuda(), tgt.cuda()
        mb, tgts = Variable(mb), Variable(tgts)
        out = model(mb)
        loss = criterion(out, tgts)
        print("Loss: {0:.05f}".format(loss.data[0]))
        optimizer.step()
