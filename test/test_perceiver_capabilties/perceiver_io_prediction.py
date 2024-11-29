import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.perceiver_core.perceiver_io import PerceiverIO


class TimeSeriesDataset(TensorDataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return (self.data[index : index + self.seq_length], self.data[index + 1 : index + self.seq_length + 1])


class PerceiverIOTimeSeriesPredictor(pl.LightningModule):
    def __init__(
        self,
        num_layers=8,
        dim=1,
        queries_dim=64,
        num_latent_tokens=64,
        latent_dimension=128,
        cross_attention_heads=4,
        latent_attention_heads=4,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        num_fourier_bands=4,
        max_frequency=10.0,
        num_input_axes=1,
        sequence_length=12,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length

        self.perceiver_io = PerceiverIO(
            num_layers=num_layers,
            dim=dim,
            queries_dim=queries_dim,
            logits_dimension=1,  # we predict a single value
            num_latent_tokens=num_latent_tokens,
            latent_dimension=latent_dimension,
            cross_attention_heads=cross_attention_heads,
            latent_attention_heads=latent_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            latent_attention_head_dim=latent_attention_head_dim,
            num_fourier_bands=num_fourier_bands,
            max_frequency=max_frequency,
            num_input_axes=num_input_axes,
            position_encoding_type="fourier",
        )

        self.query = nn.Parameter(torch.randn(1, queries_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        queries = self.query.expand(batch_size, 1, -1)
        return self.perceiver_io(x, queries=queries)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y[:, -1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y[:, -1:])
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y[:, -1:])
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def prepare_data():
    # using a very simple timeseries dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, usecols=[1], engine="python", skipfooter=3)
    data = df.values.astype("float32")

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    return data, scaler


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")


def plot_predictions(actual, predicted, future_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(range(len(actual), len(actual) + len(future_predictions)), future_predictions, label="Future Predictions")
    plt.plot(range(len(actual) - len(predicted), len(actual)), predicted, label="Model Predictions")
    plt.xlabel("Time")
    plt.ylabel("Passengers")
    plt.title("Air passengers prediction")
    plt.legend()
    plt.savefig("perceiver_io_time_series_predictions.png")


def main():
    data, scaler = prepare_data()
    sequence_length = 24  # how many months to look back to make a predicition
    dataset = TimeSeriesDataset(data, sequence_length)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = PerceiverIOTimeSeriesPredictor(sequence_length=sequence_length)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./",
        filename="perceiver-io-time-series-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=150,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # not best practice, but for the sake of this example, we will evaluate the model on the entire dataset
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(data) - sequence_length):
            input_seq = torch.FloatTensor(data[i : i + sequence_length]).unsqueeze(0)
            target = data[i + sequence_length]
            pred = model(input_seq).squeeze().item()
            all_predictions.append(pred)
            all_targets.append(target)

    # we nverse transform predictions and targets scale them back to the original values
    all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    all_targets = scaler.inverse_transform(np.array(all_targets).reshape(-1, 1))

    print("\nModel Performance on Historical Data:")
    evaluate_predictions(all_targets, all_predictions)

    # looking into the future
    future_predictions = []
    input_sequence = torch.FloatTensor(data[-sequence_length:]).unsqueeze(0)

    months_to_predict = 36  # predict the next n months

    for _ in range(months_to_predict):
        with torch.no_grad():
            pred = model(input_sequence).squeeze()
        future_predictions.append(pred.item())
        pred = pred.reshape(1, 1, 1)
        input_sequence = torch.cat([input_sequence[:, 1:], pred], dim=1)

    # inverse transform the future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    print(f"\nFuture {months_to_predict}-month predictions:")
    for i, pred in enumerate(future_predictions):
        print(f"Month {i+1}: {pred[0]:.2f}")

    actual_data = scaler.inverse_transform(data)
    plot_predictions(actual_data, all_predictions, future_predictions)


if __name__ == "__main__":
    main()
