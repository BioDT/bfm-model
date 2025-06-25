import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from bfm_model.perceiver_core.perceiver_io import PerceiverIO


class BasePerceiverIOClassifier(pl.LightningModule):
    def __init__(
        self,
        num_layers=1,
        dim=1,
        queries_dim=128,
        num_latent_tokens=32,
        latent_dimension=64,
        cross_attention_heads=1,
        latent_attention_heads=2,
        cross_attention_head_dim=32,
        latent_attention_head_dim=32,
        num_classes=10,
        num_fourier_bands=4,
        max_frequency=10.0,
        num_input_axes=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits.argmax(dim=-1), y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.permute(0, 2, 3, 1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits.argmax(dim=-1), y, task="multiclass", num_classes=self.num_classes)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


class LogitsPerceiverIOClassifier(BasePerceiverIOClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # use logits_dimension to set the number of classes
        self.perceiver_io = PerceiverIO(
            num_layers=self.hparams.num_layers,
            dim=self.hparams.dim,
            queries_dim=self.hparams.queries_dim,
            logits_dimension=self.hparams.num_classes,
            num_latent_tokens=self.hparams.num_latent_tokens,
            latent_dimension=self.hparams.latent_dimension,
            cross_attention_heads=self.hparams.cross_attention_heads,
            latent_attention_heads=self.hparams.latent_attention_heads,
            cross_attention_head_dim=self.hparams.cross_attention_head_dim,
            latent_attention_head_dim=self.hparams.latent_attention_head_dim,
            num_fourier_bands=self.hparams.num_fourier_bands,
            max_frequency=self.hparams.max_frequency,
            num_input_axes=self.hparams.num_input_axes,
            position_encoding_type="fourier",
        )

        # use a single query vector for all classes
        self.query = nn.Parameter(torch.randn(1, self.hparams.queries_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.query.expand(batch_size, -1, -1)
        return self.perceiver_io(x, queries=queries).squeeze(1)


class QueryPerceiverIOClassifier(BasePerceiverIOClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perceiver_io = PerceiverIO(
            num_layers=self.hparams.num_layers,
            dim=self.hparams.dim,
            queries_dim=self.hparams.queries_dim,
            logits_dimension=None,
            num_latent_tokens=self.hparams.num_latent_tokens,
            latent_dimension=self.hparams.latent_dimension,
            cross_attention_heads=self.hparams.cross_attention_heads,
            latent_attention_heads=self.hparams.latent_attention_heads,
            cross_attention_head_dim=self.hparams.cross_attention_head_dim,
            latent_attention_head_dim=self.hparams.latent_attention_head_dim,
            num_fourier_bands=self.hparams.num_fourier_bands,
            max_frequency=self.hparams.max_frequency,
            num_input_axes=self.hparams.num_input_axes,
            position_encoding_type="fourier",
        )

        # use multiple queries, one for each class (meaning the output of IO would be of shape [batch_size, num_classes, queries_dim])
        # we already have a separate output vector for each class
        self.class_queries = nn.Parameter(torch.randn(self.hparams.num_classes, self.hparams.queries_dim))

        # the output for each query is passed through a small MLP to to reduce each queries_dim class-specific vector to a scalar (logit)
        self.output_processor = nn.Sequential(
            nn.Linear(self.hparams.queries_dim, self.hparams.queries_dim), nn.ReLU(), nn.Linear(self.hparams.queries_dim, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        output = self.perceiver_io(x, queries=queries)
        return self.output_processor(output).squeeze(-1)


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_full = MNIST(root="./", train=True, transform=transform, download=True)
    mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
    mnist_test = MNIST(root="./", train=False, transform=transform, download=True)

    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=64, num_workers=4)

    # choose the classifier type here (i.e., one that solely uses queries only, or one that translates internally to logits)
    # model = LogitsPerceiverIOClassifier()
    model = QueryPerceiverIOClassifier()

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_acc",
    #     dirpath="./",
    #     filename="perceiver-io-mnist-{epoch:02d}-{val_acc:.2f}",
    #     save_top_k=3,
    #     mode="max",
    # )

    trainer = pl.Trainer(
        max_epochs=10,
        devices=1 if torch.cuda.is_available() else None,
        # callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
