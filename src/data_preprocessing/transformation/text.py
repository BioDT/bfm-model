# src/data_preprocessing/transformation/text.py

import warnings

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore", message=r"`clean_up_tokenization_spaces` was not set.")


def label_encode(df, column_name):
    """
    Performs label encoding on a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be encoded.
        column_name (str): The column name to be label encoded.

    Returns:
        torch.Tensor: A tensor representing the label encoded column.
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[column_name].astype(str).tolist())
    return torch.tensor(labels, dtype=torch.long)


def bert_tokenizer(corpus: list, max_length: int, batch_size: int = 16, device: str = "cpu") -> torch.Tensor:
    """
    Generates BERT embeddings for a given corpus of text.

    Args:
        corpus (list): A list of strings where each element is a text sample
                        to be embedded using the BERT model.
        max_length (int): The maximum length for tokenization.
        batch_size (int, optional): The number of samples to process in each batch. Default is 16.
        device (str, optional): The device to run the model on, either "cpu" or "cuda"
                                for GPU acceleration. Default is "cpu".

    Returns:
        torch.Tensor: A tensor containing the BERT embeddings for each input text in the corpus.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(device)

    bert_embeddings = []

    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        bert_embeddings.append(batch_embeddings)

    bert_embeddings = torch.cat(bert_embeddings).cpu()

    return bert_embeddings


def pad_or_truncate_embeddings(embeddings: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pads or truncates BERT embeddings to a fixed length (target_length).

    Args:
        embeddings (torch.Tensor): The BERT embeddings tensor of shape (N, 768), where N is the number of texts.
        target_length (int): The target length (number of texts) to pad or truncate the embeddings to.

    Returns:
        torch.Tensor: The padded or truncated embeddings tensor of shape (target_length, 768).
    """
    current_length = embeddings.size(0)

    if current_length < target_length:
        padding_size = target_length - current_length
        padding = torch.zeros(padding_size, embeddings.size(1))
        embeddings = torch.cat([embeddings, padding], dim=0)
    elif current_length > target_length:
        embeddings = embeddings[:target_length]

    return embeddings


def reduce_embedding_dimensions(embeddings: torch.Tensor, output_dim: int = 128) -> torch.Tensor:
    """
    Reduces the dimensionality of BERT embeddings using PCA.

    Args:
        embeddings (torch.Tensor): The BERT embeddings tensor of shape (N, 768).
        output_dim (int): The target dimensionality for the embeddings (default is 128).

    Returns:
        torch.Tensor: The dimensionality-reduced embeddings tensor of shape (N, output_dim).
    """
    pca = PCA(n_components=output_dim)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    return torch.tensor(reduced_embeddings)
