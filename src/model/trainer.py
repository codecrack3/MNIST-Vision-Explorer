"""Model training utilities and CLI entrypoint."""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model.network import ConvNet


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    data_dir: Path = Path("data")
    weights_path: Path = Path("src/model/weights/mnist_cnn.pt")
    device: str | None = None


@dataclass(frozen=True)
class TrainProgress:
    state: str
    epoch: int
    total_epochs: int
    train_loss: float | None
    val_accuracy: float | None
    error: str | None = None


@dataclass(frozen=True)
class TrainResult:
    success: bool
    final_accuracy: float | None
    weights_path: Path | None
    error: str | None = None


StatusCallback = Callable[[TrainProgress], None]


def _resolve_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(root=str(config.data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(config.data_dir), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    return train_loader, test_loader


def _evaluate_accuracy(model: ConvNet, test_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += int((predictions == labels).sum().item())

    if total == 0:
        return 0.0
    return correct / total


def _save_weights_atomic(model: ConvNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp_path)
    tmp_path.replace(path)


def train_model(
    config: TrainConfig,
    status_cb: StatusCallback | None = None,
    *,
    stop_event: threading.Event | None = None,
) -> TrainResult:
    """Train a model and persist weights."""

    def emit(progress: TrainProgress) -> None:
        if status_cb is not None:
            status_cb(progress)

    device = _resolve_device(config.device)

    try:
        train_loader, test_loader = _build_loaders(config)

        model = ConvNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = nn.CrossEntropyLoss()

        emit(
            TrainProgress(
                state="training",
                epoch=0,
                total_epochs=config.epochs,
                train_loss=None,
                val_accuracy=None,
            )
        )

        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            batches = 0

            for images, labels in train_loader:
                if stop_event is not None and stop_event.is_set():
                    return TrainResult(success=False, final_accuracy=None, weights_path=None, error="training_stopped")

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                batches += 1

            average_loss = running_loss / max(batches, 1)
            val_accuracy = _evaluate_accuracy(model, test_loader, device)

            emit(
                TrainProgress(
                    state="training",
                    epoch=epoch,
                    total_epochs=config.epochs,
                    train_loss=average_loss,
                    val_accuracy=val_accuracy,
                )
            )

        model = model.cpu()
        _save_weights_atomic(model, config.weights_path)

        emit(
            TrainProgress(
                state="ready",
                epoch=config.epochs,
                total_epochs=config.epochs,
                train_loss=average_loss,
                val_accuracy=val_accuracy,
            )
        )

        return TrainResult(success=True, final_accuracy=val_accuracy, weights_path=config.weights_path)

    except Exception as exc:  # pragma: no cover - defensive path
        error_message = str(exc)
        emit(
            TrainProgress(
                state="failed",
                epoch=0,
                total_epochs=config.epochs,
                train_loss=None,
                val_accuracy=None,
                error=error_message,
            )
        )
        return TrainResult(success=False, final_accuracy=None, weights_path=None, error=error_message)


def run_training_cli() -> None:
    """CLI entrypoint for offline model training."""
    parser = argparse.ArgumentParser(description="Train the MNIST ConvNet and save weights.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--weights-path", type=Path, default=Path("src/model/weights/mnist_cnn.pt"))
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        weights_path=args.weights_path,
        device=args.device,
    )

    def printer(progress: TrainProgress) -> None:
        if progress.state == "training" and progress.epoch > 0:
            print(
                f"epoch={progress.epoch}/{progress.total_epochs} "
                f"loss={progress.train_loss:.4f} val_acc={progress.val_accuracy:.4f}"
            )
        elif progress.state == "failed":
            print(f"training failed: {progress.error}")

    result = train_model(config, status_cb=printer)
    if not result.success:
        raise SystemExit(1)

    print(f"training complete: accuracy={result.final_accuracy:.4f} weights={result.weights_path}")
