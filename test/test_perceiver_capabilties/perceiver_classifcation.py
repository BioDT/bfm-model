import glob
import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from bfm_model.perceiver_core.perceiver_original import Perceiver


class PerceiverClassifier(pl.LightningModule):
    """
    Test class for a Perceiver model that can be used for classification tasks, such as MNIST.
    """

    def __init__(
        self,
        num_fourier_bands=6,
        num_layers=1,
        max_frequency=10.0,
        input_channels=1,
        num_input_axes=2,
        num_latent_tokens=256,
        latent_dimension=512,
        cross_attention_heads=1,
        self_attention_heads=8,
        cross_attention_head_dim=64,
        self_attention_head_dim=64,
        num_classes=10,
        attention_dropout=0.0,
        feedforward_dropout=0.0,
        weight_tie_layers=True,
        use_fourier_encoding=True,
        self_attentions_per_cross=2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        # initialize the Perceiver model, nothing else!
        self.perceiver = Perceiver(
            num_fourier_bands=num_fourier_bands,
            num_layers=num_layers,
            max_frequency=max_frequency,
            input_channels=input_channels,
            num_input_axes=num_input_axes,
            num_latent_tokens=num_latent_tokens,
            latent_dimension=latent_dimension,
            cross_attention_heads=cross_attention_heads,
            self_attention_heads=self_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            self_attention_head_dim=self_attention_head_dim,
            num_classes=num_classes,
            attention_dropout=attention_dropout,
            feedforward_dropout=feedforward_dropout,
            weight_tie_layers=weight_tie_layers,
            use_fourier_encoding=use_fourier_encoding,
            self_attentions_per_cross=self_attentions_per_cross,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.perceiver(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step on a batch of data.

        :param batch: The batch of data to train on.
        :param batch_idx: The index of the batch.

        :return: The loss of the training step.
        """
        x, y = batch
        x = x.permute(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)
        prediction = self(x)

        loss = F.cross_entropy(prediction, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Self-explanatory, perform a validation step.

        :param batch: The batch of data to validate on.
        :param batch_idx: The index of the batch.

        :return: The loss and accuracy of the validation step.
        """
        x, y = batch
        x = x.permute(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)
        prediction = self(x)

        loss = F.cross_entropy(prediction, y)
        acc = accuracy(prediction.argmax(dim=-1), y, task="multiclass", num_classes=self.num_classes)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a test step on a batch of data.

        :param batch: The batch of data to test on.
        :param batch_idx: The index of the batch.

        :return: The loss and accuracy of the test step.
        """
        x, y = batch
        x = x.permute(0, 2, 3, 1)  # Change from (B, C, H, W) to (B, H, W, C)
        prediction = self(x)

        loss = F.cross_entropy(prediction, y)
        acc = accuracy(prediction.argmax(dim=-1), y, task="multiclass", num_classes=self.num_classes)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Configure the optimizer and scheduler for the model.

        :return: The optimizer and scheduler for the model.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)  # using the AdamW optimizer, for proper weight decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # using an exponential learning rate scheduler
        return [optimizer], [scheduler]


def main():
    # define the preprocessing steps for the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]  # MNIST mean and std, after normalization
    )

    # load the MNIST dataset and split it into training, validation, and test sets
    mnist_full = MNIST(root="./", train=True, transform=transform, download=True)
    mnist_train, mnist_val = random_split(mnist_full, [55000, 5000])
    mnist_test = MNIST(root="./", train=False, transform=transform, download=True)

    # create the data loaders for the training, validation, and test sets
    train_loader = DataLoader(mnist_train, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, num_workers=4)
    test_loader = DataLoader(mnist_test, batch_size=64, num_workers=4)

    model = PerceiverClassifier(
        num_fourier_bands=4,
        num_layers=1,
        max_frequency=10.0,
        input_channels=1,
        num_input_axes=2,
        num_latent_tokens=32,
        latent_dimension=64,
        cross_attention_heads=1,
        self_attention_heads=2,
        cross_attention_head_dim=32,
        self_attention_head_dim=32,
        num_classes=10,
        attention_dropout=0.0,
        feedforward_dropout=0.0,
        weight_tie_layers=True,
    )

    # create a ModelCheckpoint callback to save the best model(s) during training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="./",
        filename="perceiver-mnist-{epoch:02d}-{val_acc:.2f}",
        save_top_k=3,
        mode="max",
    )

    # create a PyTorch Lightning Trainer and train the model
    trainer = pl.Trainer(
        max_epochs=2,
        devices=1 if torch.cuda.is_available() else None,  # use GPU if available
        callbacks=[checkpoint_callback],
    )

    # train and test the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # delete MNIST dataset from the repository, and the checkpoints and logs
    import shutil

    shutil.rmtree("./MNIST", ignore_errors=True)
    shutil.rmtree("./lightning_logs", ignore_errors=True)
    checkpoints = glob.glob("perceiver-mnist*")
    for checkpoint in checkpoints:
        os.remove(checkpoint)


if __name__ == "__main__":
    main()

# TODO:
# 1) Add functionality to upport customizing the testing procedure with Hydra
