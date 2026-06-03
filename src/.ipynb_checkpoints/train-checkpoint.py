import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import BrainTumorCNN, build_resnet18

TRAIN_DIR   = '../data/Training'
TEST_DIR    = '../data/Testing'
RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            correct  += out.argmax(1).eq(labels).sum().item()
            total    += labels.size(0)
    return loss_sum / total, correct / total


def train(model, model_name, train_loader, val_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    history   = {'train_loss': [], 'val_loss': [],
                 'train_acc':  [], 'val_acc':  []}
    best_acc  = 0.0

    print(f"\n{'='*60}")
    print(f" Training: {model_name} | Device: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, f'{model_name}_best.pth'))
            flag = ' ✓ saved'
        else:
            flag = ''

        print(f"Ep {epoch+1:02d}/{epochs} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Acc {tr_acc*100:.1f}%/{vl_acc*100:.1f}% | "
              f"{time.time()-t0:.1f}s{flag}")

    print(f"\nBest val accuracy: {best_acc*100:.2f}%")
    return history


def plot_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history['train_loss'], 'b-o', ms=4, label='Train')
    ax1.plot(epochs, history['val_loss'],   'r-o', ms=4, label='Val')
    ax1.set(title=f'{model_name} — Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'b-o', ms=4, label='Train')
    ax2.plot(epochs, [a*100 for a in history['val_acc']],   'r-o', ms=4, label='Val')
    ax2.set(title=f'{model_name} — Accuracy', xlabel='Epoch', ylabel='Accuracy (%)')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'{model_name.lower().replace(" ","_")}_curves.png'),
                dpi=150, bbox_inches='tight')
    print(f"Saved training curves!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str,   default='resnet',
                        choices=['cnn', 'resnet'])
    parser.add_argument('--epochs', type=int,   default=15)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--batch',  type=int,   default=32)
    args = parser.parse_args()

    train_loader, val_loader, _, _ = get_dataloaders(
        TRAIN_DIR, TEST_DIR, batch_size=args.batch)

    if args.model == 'cnn':
        model      = BrainTumorCNN(num_classes=4).to(device)
        model_name = 'Custom_CNN'
    else:
        model      = build_resnet18(num_classes=4).to(device)
        model_name = 'ResNet18_Transfer'

    history = train(model, model_name, train_loader, val_loader,
                    epochs=args.epochs, lr=args.lr)
    plot_history(history, model_name)


if __name__ == '__main__':
    main()