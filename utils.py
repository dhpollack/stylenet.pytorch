import torch
import csv
import shutil

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_loss_csv(losses, filename="train_losses.csv"):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(losses)
