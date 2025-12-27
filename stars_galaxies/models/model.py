import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class GalaxyStarClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 16 * 16, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        # Метрики
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2)
        self.val_roc_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)

        # Для накопления результатов на эпоху
        self._val_logits = []
        self._val_targets = []

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # сохраняем логиты для AUROC
        self._val_logits.append(logits)
        self._val_targets.append(y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        logits = torch.cat(self._val_logits)  # [N, num_classes]
        targets = torch.cat(self._val_targets)  # [N]

        preds = torch.argmax(logits, dim=1)

        f1 = self.val_f1(preds, targets)  # на метках
        auc = self.val_roc_auc(logits, targets)  # на логитах

        self.log("val_f1", f1, prog_bar=True)
        self.log("val_roc_auc", auc, prog_bar=True)

        # очищаем списки для следующей эпохи
        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
