import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    """
    Store input token windows and their next-token prediction targets
    """

    def __init__(self, tokenizer, txt, sample_Lenth, stride):
        """
        Encode text and create input/target pairs using a sliding window.
        """
        self.input_ids = []
        self.target_ids = []

        # tokenize
        token_ids = tokenizer.encode(txt)
        if len(token_ids) <= sample_Lenth:
            raise ValueError("Token length is too short")

        # sliding window to have a sequence for prediction
        for i in range(0, len(token_ids) - sample_Lenth, stride):
            input_chunk = token_ids[i:i + sample_Lenth]
            target_chunk = token_ids[i + 1 : i + sample_Lenth + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def createDataLoader(txt, batch_size=4, seq_Lenth=1024,
                     stride=256, shuffle=True, drop_last=True,
                     num_workers=0):
    """Build a DataLoader for next-token prediction using GPT-2 tokenization

    txt - Source text.
    batch_size - Sequences per batch.
    seq_Lenth - Tokens per sequence.
    stride - Tokens between window starts.
    shuffle - Randomize sample order.
    drop_last - Discard the incomplete final batch.
    num_workers - Data-loading worker processes.

    Batch shape - Two tensors of shape (current_batch_size, seq_Lenth).
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(tokenizer, txt, seq_Lenth, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
