import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_dataloaders
from model import BrainTumorCNN, build_resnet18

TRAIN_DIR   = '../data/Training'
TEST_DIR    = '../data/Testing'
RESULTS_DIR = '../results'
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs  = imgs.to(device)
            out   = model(imgs)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                linewidths=0.5)
    plt.title(f'{model_name} — Confusion Matrix',
              fontweight='bold', fontsize=13)
    plt.ylabel('True Label',      fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'{model_name.lower().replace(" ","_")}_confusion.png'),
                dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved!")
    plt.show()


def visualise_predictions(model, loader, model_name, n=8):
    model.eval()
    imgs_list, preds_list, labels_list = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out   = model(imgs.to(device))
            preds = out.argmax(1).cpu().numpy()
            imgs_list.extend(imgs.cpu())
            preds_list.extend(preds)
            labels_list.extend(labels.numpy())
            if len(imgs_list) >= n:
                break

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(f'{model_name} — Sample Predictions',
                 fontsize=13, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        img   = imgs_list[i].permute(1, 2, 0).numpy()
        img   = np.clip(img * std + mean, 0, 1)
        pred  = CLASS_NAMES[preds_list[i]]
        true  = CLASS_NAMES[labels_list[i]]
        color = 'green' if preds_list[i] == labels_list[i] else 'red'
        ax.imshow(img)
        ax.set_title(f'Pred: {pred}\nTrue: {true}',
                     color=color, fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'{model_name.lower().replace(" ","_")}_predictions.png'),
                dpi=150, bbox_inches='tight')
    print(f"Predictions saved!")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, required=True,
                        choices=['cnn', 'resnet'])
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()

    _, _, test_loader, _ = get_dataloaders(TRAIN_DIR, TEST_DIR)

    if args.model == 'cnn':
        model      = BrainTumorCNN(num_classes=4).to(device)
        model_name = 'Custom CNN'
    else:
        model      = build_resnet18(num_classes=4,
                                    freeze_backbone=False).to(device)
        model_name = 'ResNet18 Transfer'

    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    labels, preds = get_predictions(model, test_loader)
    acc = (labels == preds).mean()

    print(f"\n{'='*50}")
    print(f" {model_name} — Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    plot_confusion_matrix(labels, preds, model_name)
    visualise_predictions(model, test_loader, model_name)


if __name__ == '__main__':
    main()