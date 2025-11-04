import torch
import torch.nn as nn
import timm
from timm.data import create_loader, resolve_model_data_config
from tqdm import tqdm
import argparse


class LinearProbModel(nn.Module):
    """DINOv3 backbone with frozen weights + linear classifier"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_datasets_and_loaders(model, args):
    """Create datasets and dataloaders using timm"""
    # Create datsets
    train_dataset = timm.data.create_dataset(
        name=f'torch/{args.dataset}',
        root=args.data_dir,
        split='train',
        download=True,
    )

    val_dataset = timm.data.create_dataset(
        name=f'torch/{args.dataset}',
        root=args.data_dir,
        split='validation',
        download=True,
    )

    # Get data config from model
    data_config = resolve_model_data_config(model)

    # Create loaders
    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        pin_memory=True,
        num_workers=args.num_workers,
        **data_config
    )

    val_loader = create_loader(
        val_dataset,
        batch_size=args.batch_size * 2,
        is_training=False,
        use_prefetcher=True,
        pin_memory=True,
        num_workers=args.num_workers,
        **data_config
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}'
        })

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Validation')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load DINOv3 backbone
    print(f'Loading model: {args.model}')
    backbone = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=0  # Remove classification head
    )

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = LinearProbModel(backbone, num_classes).to(device)

    print(f'Model features: {backbone.num_features}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.classifier.parameters())}')

    # Create datasets and loaders
    train_loader, val_loader = create_datasets_and_loaders(backbone, args)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'\nEpoch {epoch}/{args.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}\n')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, args.checkpoint_path)
            print(f'  -> Saved best model (Val Acc: {best_acc:.2f}%)\n')

    print(f'Training finished! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear probin on DINOv3')

    # Model
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_dinov3.lvd1689m',
                        help='DINOv3 model variant')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for dataset')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--checkpoint-path', type=str,
                        default='best_model.pth',
                        help='Path to save best model')

    args = parser.parse_args()
    main(args)