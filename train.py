"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.databuild import AnimalDataset
from src.cnn import CNN
from src.utils import *
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from torch.optim import SGD, Adagrad, RMSprop, Adam
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import tensorboard

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
    ])

    train_set = AnimalDataset(root=args.data_path, train=True, transform=transform)
    valid_set = AnimalDataset(root=args.data_path2, train=False, transform=transform)

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": 4,
    }

    valid_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 4,
    }

    training_dataloader = DataLoader(train_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **valid_params)

    model = CNN(num_classes=len(train_set.animalsList)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter(args.tensorboard_path)

    total_iters = len(training_dataloader)
    for epoch in range(start_epoch, args.epochs):
        # Training Phase
        model.train()
        losses = []
        progress_bar = tqdm(training_dataloader, colour='cyan')
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            loss = criterion(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            progress_bar.set_description("Epoch {}/{}. Loss value: {:.4f}".format(epoch+1, args.epochs, loss_val))
            losses.append(loss_val)
            writer.add_scalar("Train/Loss", np.mean(losses), epoch*total_iters+iter)

        # Valid Phase
        model.eval()
        losss = []
        all_predictions = []
        all_ground_truth = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(valid_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                prediction = model(images)
                maxVal_ofIdx = torch.argmax(prediction, dim=1)
                loss = criterion(prediction, labels)
                losss.append(loss.item())
                all_ground_truth.extend(labels.tolist())
                all_predictions.extend(maxVal_ofIdx.tolist())

        writer.add_scalar("Validation/Loss", np.mean(losss), epoch)
        acc = accuracy_score(all_ground_truth, all_predictions)
        writer.add_scalar("Validation/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.animalsList))], epoch)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "batch_size": args.batch_size,
        }

        torch.save(checkpoint, os.path.join(args.trained_models, "last.pth"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.trained_models, "best.pth"))
            best_acc = acc
        scheduler.step()

if __name__ == '__main__':
    args = get_args()
    train(args)
