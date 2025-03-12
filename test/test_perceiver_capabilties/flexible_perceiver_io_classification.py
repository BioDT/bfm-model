# TODO: Work in progress..
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from bfm_model.perceiver_core.flexible_perceiver_io import FlexiblePerceiverIO


class JointClassifier(pl.LightningModule):
    def __init__(
        self,
        input_configs,
        num_layers=6,
        num_latent_token=256,
        latent_dimension=512,
        cross_attention_heads=2,
        latent_attention_heads=8,
        cross_attention_head_dim=64,
        latent_attention_head_dim=64,
        queries_dim=256,
        image_classes=10,
        text_classes=3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FlexiblePerceiverIO(
            input_configs=input_configs,
            num_layers=num_layers,
            num_latent_tokens=num_latent_token,
            latent_dimension=latent_dimension,
            cross_attention_heads=cross_attention_heads,
            latent_attention_heads=latent_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            latent_attention_head_dim=latent_attention_head_dim,
            queries_dim=queries_dim,
            logits_dimension=None,  # queries all the way
        )

        # separate queries for each task
        self.image_query = nn.Parameter(torch.randn(1, queries_dim))
        self.text_query = nn.Parameter(torch.randn(1, queries_dim))

        # output projections for each task
        self.image_projection = nn.Linear(queries_dim, image_classes)
        self.text_projection = nn.Linear(queries_dim, text_classes)

        self.image_loss_fn = nn.CrossEntropyLoss()
        self.text_loss_fn = nn.CrossEntropyLoss()

    def forward(self, image, text):
        batch_size = image.shape[0]

        # prep the queries
        image_queries = self.image_query.expand(batch_size, -1, -1)
        text_queries = self.text_query.expand(batch_size, -1, -1)
        queries = torch.cat([image_queries, text_queries], dim=1)

        # the forward pass through our flexible perceiver
        outputs = self.model({"image": image, "text": text}, queries=queries)

        # split the outputs for each task
        image_out, text_out = outputs.chunk(2, dim=1)

        # to logits
        image_logits = self.image_projection(image_out.squeeze(1))
        text_logits = self.text_projection(text_out.squeeze(1))

        return image_logits, text_logits

    def training_step(self, batch, batch_idx):
        (image, image_label), (text, text_label) = batch
        # image = image.permute(0, 2, 3, 1)
        image_logits, text_logits = self(image, text)

        image_loss = self.image_loss_fn(image_logits, image_label)
        text_loss = self.text_loss_fn(text_logits, text_label)

        # combine losses (maybe add some weight as well?)
        total_loss = image_loss + text_loss

        self.log("train_image_loss", image_loss)
        self.log("train_text_loss", text_loss)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_loss(self, batch, batch_idx):
        (image, image_label), (text, text_label) = batch
        # image = image.permute(0, 2, 3, 1)
        image_logits, text_logits = self(image, text)

        image_loss = self.image_loss_fn(image_logits, image_label)
        text_loss = self.text_loss_fn(text_logits, text_label)

        # combine losses (maybe add some weight as well?)
        total_loss = image_loss + text_loss

        image_acc = (image_logits.argmax(dim=-1) == image_label).float().mean()
        text_acc = (text_logits.argmax(dim=-1) == text_label).float().mean()

        self.log("val_image_loss", image_loss)
        self.log("val_text_loss", text_loss)
        self.log("val_total_loss", total_loss)
        self.log("val_image_acc", image_acc)
        self.log("val_text_acc", text_acc)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_data, text_data, text_labels):
        self.mnist_data = mnist_data
        self.text_data = torch.FloatTensor(text_data).unsqueeze(1)  # Add sequence dimension
        self.text_labels = torch.LongTensor(text_labels)

    def __len__(self):
        return min(len(self.mnist_data), len(self.text_data))

    def __getitem__(self, idx):
        mnist_item = self.mnist_data[idx]
        return (mnist_item[0].view(-1, 1), mnist_item[1]), (self.text_data[idx], self.text_labels[idx])


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = MNIST(root="./", train=True, download=True, transform=transform)
    mnist_val = MNIST(root="./", train=False, transform=transform)

    newsgroups = fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space", "rec.sport.baseball"])
    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    x = vectorizer.fit_transform(newsgroups.data).toarray()
    y = newsgroups.target

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = JointDataset(mnist_train, x_train, y_train)
    val_dataset = JointDataset(mnist_val, x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    input_configs = {
        "image": {"dim": 1, "max_len": 784},  # MNIST: 28x28 = 784
        "text": {"dim": 1000, "max_len": 1},  # Text: 1000-dim bag-of-words vector, 1 sequence length
    }

    model = JointClassifier(input_configs)

    trainer = pl.Trainer(
        max_epochs=10,
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
