"""Application-level model lifecycle and training state."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model.inference import extract_activations, predict
from src.model.network import ConvNet
from src.model.trainer import TrainConfig, TrainProgress, train_model
from src.visualization.filters import get_filter_payload
from src.visualization.weights import get_weight_payload


@dataclass(frozen=True)
class StartTrainingResult:
    accepted: bool
    job_id: str | None = None
    state: str | None = None
    reason: str | None = None


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class ModelState:
    """Holds the loaded model and mutable training metadata."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        self.weights_path = Path(weights_path or "src/model/weights/mnist_cnn.pt")

        self.model: ConvNet | None = None
        self.model_version = "cnn-v1"
        self._current_job_id: str | None = None
        self._training_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._status: dict[str, Any] = {
            "state": "untrained",
            "epoch": 0,
            "total_epochs": 0,
            "train_loss": None,
            "val_accuracy": None,
            "last_error": None,
            "weights_path": str(self.weights_path),
            "updated_at": _utc_timestamp(),
        }

        self._load_weights_if_available()

    def _set_status_locked(
        self,
        *,
        state: str,
        epoch: int | None = None,
        total_epochs: int | None = None,
        train_loss: float | None = None,
        val_accuracy: float | None = None,
        last_error: str | None = None,
    ) -> None:
        self._status["state"] = state
        if epoch is not None:
            self._status["epoch"] = epoch
        if total_epochs is not None:
            self._status["total_epochs"] = total_epochs
        self._status["train_loss"] = train_loss
        self._status["val_accuracy"] = val_accuracy
        self._status["last_error"] = last_error
        self._status["updated_at"] = _utc_timestamp()

    def _load_model_from_path(self, path: Path) -> ConvNet:
        model = ConvNet()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_weights_if_available(self) -> None:
        if not self.weights_path.exists():
            return

        try:
            model = self._load_model_from_path(self.weights_path)
        except Exception as exc:  # pragma: no cover - defensive path
            with self._lock:
                self.model = None
                self._set_status_locked(
                    state="failed",
                    epoch=0,
                    total_epochs=0,
                    train_loss=None,
                    val_accuracy=None,
                    last_error=f"Failed to load weights: {exc}",
                )
            return

        with self._lock:
            self.model = model
            self._set_status_locked(
                state="ready",
                epoch=0,
                total_epochs=0,
                train_loss=None,
                val_accuracy=None,
                last_error=None,
            )

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
        status["model_version"] = self.model_version
        return status

    def set_status_for_testing(
        self,
        *,
        state: str,
        epoch: int,
        total_epochs: int,
        train_loss: float | None,
        val_accuracy: float | None,
        last_error: str | None,
    ) -> None:
        with self._lock:
            self._set_status_locked(
                state=state,
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=train_loss,
                val_accuracy=val_accuracy,
                last_error=last_error,
            )

    def _require_model(self) -> ConvNet:
        with self._lock:
            model = self.model

        if model is None:
            raise RuntimeError(
                f"Model is not loaded. Train a model first or provide weights at {self.weights_path}."
            )
        return model

    def predict(self, image_28x28: np.ndarray) -> dict[str, Any]:
        model = self._require_model()
        result = predict(model, image_28x28, model_version=self.model_version)
        return {
            "predicted_class": result.predicted_class,
            "probabilities": result.probabilities,
            "confidence": result.confidence,
            "latency_ms": round(result.latency_ms, 4),
            "model_version": result.model_version,
        }

    def activations(self, image_28x28: np.ndarray) -> dict[str, Any]:
        model = self._require_model()
        result = extract_activations(model, image_28x28)
        return {
            "conv1": result.conv1,
            "conv2": result.conv2,
            "fc1": result.fc1,
            "output_probs": result.output_probs,
        }

    def filters(self) -> dict[str, Any]:
        model = self._require_model()
        return get_filter_payload(model)

    def weights(self) -> dict[str, Any]:
        model = self._require_model()
        return get_weight_payload(model)

    def _progress_callback(self, progress: TrainProgress, *, job_id: str) -> None:
        with self._lock:
            if self._current_job_id != job_id:
                return
            self._set_status_locked(
                state=progress.state,
                epoch=progress.epoch,
                total_epochs=progress.total_epochs,
                train_loss=progress.train_loss,
                val_accuracy=progress.val_accuracy,
                last_error=progress.error,
            )

    def _training_worker(self, config: TrainConfig, *, job_id: str, stop_event: threading.Event) -> None:
        result = train_model(
            config,
            status_cb=lambda progress: self._progress_callback(progress, job_id=job_id),
            stop_event=stop_event,
        )

        with self._lock:
            if self._current_job_id != job_id:
                return

        if result.success and result.weights_path is not None:
            try:
                model = self._load_model_from_path(result.weights_path)
            except Exception as exc:  # pragma: no cover - defensive path
                with self._lock:
                    if self._current_job_id != job_id:
                        return
                    self._set_status_locked(
                        state="failed",
                        epoch=0,
                        total_epochs=config.epochs,
                        train_loss=None,
                        val_accuracy=None,
                        last_error=f"Trained weights could not be loaded: {exc}",
                    )
                return

            with self._lock:
                if self._current_job_id != job_id:
                    return
                self.model = model
                self._set_status_locked(
                    state="ready",
                    epoch=config.epochs,
                    total_epochs=config.epochs,
                    train_loss=self._status.get("train_loss"),
                    val_accuracy=result.final_accuracy,
                    last_error=None,
                )
            return

        with self._lock:
            if self._current_job_id != job_id:
                return

            fallback_state = "ready" if self.model is not None else "untrained"
            if result.error == "training_stopped":
                self._set_status_locked(
                    state=fallback_state,
                    epoch=0,
                    total_epochs=config.epochs,
                    train_loss=None,
                    val_accuracy=None,
                    last_error="training_stopped",
                )
            else:
                self._set_status_locked(
                    state="failed",
                    epoch=0,
                    total_epochs=config.epochs,
                    train_loss=None,
                    val_accuracy=None,
                    last_error=result.error or "Unknown training error.",
                )

    def start_training(
        self,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        force_restart: bool,
    ) -> StartTrainingResult:
        existing_thread: threading.Thread | None = None
        existing_stop_event: threading.Event | None = None

        with self._lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                if not force_restart:
                    return StartTrainingResult(
                        accepted=False,
                        reason="Training is already in progress.",
                    )
                existing_thread = self._training_thread
                existing_stop_event = self._stop_event

        if existing_thread is not None and existing_stop_event is not None:
            existing_stop_event.set()
            existing_thread.join(timeout=15)
            if existing_thread.is_alive():
                return StartTrainingResult(
                    accepted=False,
                    reason="Unable to stop current training job.",
                )

        with self._lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                return StartTrainingResult(
                    accepted=False,
                    reason="Training is already in progress.",
                )

            job_id = f"local-{int(time.time())}"
            config = TrainConfig(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weights_path=self.weights_path,
            )

            self._stop_event = threading.Event()
            self._current_job_id = job_id
            self._set_status_locked(
                state="training",
                epoch=0,
                total_epochs=epochs,
                train_loss=None,
                val_accuracy=None,
                last_error=None,
            )

            thread = threading.Thread(
                target=self._training_worker,
                kwargs={"config": config, "job_id": job_id, "stop_event": self._stop_event},
                daemon=True,
                name=f"train-{job_id}",
            )
            self._training_thread = thread
            thread.start()

        return StartTrainingResult(accepted=True, job_id=job_id, state="training")
