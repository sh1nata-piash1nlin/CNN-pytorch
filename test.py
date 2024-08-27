import torch
import torch.nn as nn
import cv2
from src.cnn import CNN
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_args():
    parse = argparse.ArgumentParser(description='Football Jerseys')
    parse.add_argument('-p', '--path', type=str, default='./data/animals/animals_train')
    parse.add_argument('-p2', '--test_path', type=str, default='image_testing/dog1.jpg' )
    parse.add_argument('-i', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default="trained_models/best.pt")
    args = parse.parse_args()
    return args

def test(args):

    animalsList = []
    for filename in os.listdir(args.path):
        if os.path.isdir(os.path.join(args.path, filename)):
            animalsList.append(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=len(animalsList)).to(device)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print("A checkpoint must be provided")
        exit(0)

    if not args.test_path:
        print("An image path must be provided")
        exit(0)

    image = cv2.imread(args.test_path)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))
    image = image / 255
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float().to(device).float()
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
    probs = softmax(prediction)
    max_value, max_idx = torch.max(probs, dim=1)
    print("The pic is about {} with probability {}".format(animalsList[max_idx], max_value.item()))

    fig, ax = plt.subplots(figsize=(20, 10))
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:brown', 'tab:cyan']
    ax.bar(animalsList, probs[0].cpu().numpy(), color=bar_colors)
    ax.set_ylabel('Probability')
    ax.set_xlabel('Animal')
    ax.set_title('Animal Prediction')
    plt.show()



if __name__ == '__main__':
    args = get_args()
    test(args)















