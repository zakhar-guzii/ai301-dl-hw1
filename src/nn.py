import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import torchmetrics as tm
from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import grad_norm

from src.tracking import EXPERIMENT_NAME, TRACKING_URI

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TimeLinear(nn.Module):
    def __init__(self, steps=80):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(steps, steps))
        self.bias = nn.Parameter(torch.zeros(steps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x):
        return self.weight @ x + self.bias[:, None]


class Block(nn.Module):
    def __init__(self, d=128, dropout=0.1, time_mixing=True):
        super().__init__()
        self.time_mixing = time_mixing
        self.norm = nn.LayerNorm(d)
        self.time = TimeLinear()
        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )

    def forward(self, x):
        if self.time_mixing:
            x = x + self.drop(torch.relu(self.time(self.norm(x))))
        return x + self.net(x)


class PressureNet(nn.Module):
    def __init__(self, n_features, d=128, n_blocks=6, dropout=0.1, time_mixing=True):
        super().__init__()
        self.inp = nn.Linear(n_features, d)
        self.blocks = nn.Sequential(
            *[Block(d, dropout, time_mixing) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        x = self.inp(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        return x.squeeze(-1)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)  # m = β1·m + (1−β1)·g
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # v = β2·v + (1−β2)·g²

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                p.addcdiv_(
                    m_hat, v_hat.sqrt() + eps, value=-lr
                )  # p = p − lr·m̂/(√v̂+ε)

        return loss

class LitPressure(L.LightningModule):
    def __init__(self, n_features, d=128, n_blocks=6, dropout=0.1, time_mixing=True, lr=1e-3, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = PressureNet(n_features, d, n_blocks, dropout, time_mixing)
        self.val_mae = tm.MeanAbsoluteError()

    def training_step(self, batch, batch_idx):          
        pred = self.model(batch["features"])
        mask = batch["mask"]
        loss = F.l1_loss(pred[mask], batch["target"][mask])
        self.log("train_mae", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch["features"])
        mask = batch["mask"]
        self.val_mae.update(pred[mask], batch["target"][mask])
        self.log("val_mae", self.val_mae, prog_bar=True)

    def configure_optimizers(self):
        decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]

        opt = AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grad_norm(self.model, norm_type=2))

    def predict_step(self, batch, batch_idx):
        return self.model(batch["features"])


def train(train_loader, val_loader, n_features, run_name, max_epochs=40, **hparams):
    L.seed_everything(42)
    lit = LitPressure(n_features, **hparams)
    ckpt = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename=run_name + "-{epoch:02d}-{val_mae:.4f}",
        monitor="val_mae",
        mode="min",
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=1.0,
        logger=MLFlowLogger(
            experiment_name=EXPERIMENT_NAME,
            tracking_uri=TRACKING_URI,
            run_name=run_name,
        ),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(lit, train_loader, val_loader)
    return LitPressure.load_from_checkpoint(ckpt.best_model_path), ckpt.best_model_score.item()
