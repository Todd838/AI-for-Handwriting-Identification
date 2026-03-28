"""Torch/torchvision pieces for training and embedding (keeps data_anyscript.py importable without torch)."""

import random
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_anyscript import PageRecord


def default_transform(image_size: int = 448):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class TripletPageDataset(Dataset):
    def __init__(
        self,
        by_author: Dict[str, List[PageRecord]],
        transform=None,
        steps_per_epoch: int = 100000,
    ):
        self.by_author = by_author
        self.authors = list(by_author.keys())
        self.transform = transform or default_transform()
        self.steps_per_epoch = steps_per_epoch
        if len(self.authors) < 2:
            raise ValueError("Need at least 2 authors to sample triplets.")

    def __len__(self):
        return self.steps_per_epoch

    def _load(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        del idx
        author = random.choice(self.authors)
        positives = self.by_author[author]
        anchor_rec = random.choice(positives)
        positive_rec = random.choice(positives)

        if len(positives) > 1:
            while positive_rec.page_path == anchor_rec.page_path:
                positive_rec = random.choice(positives)

        neg_author = random.choice(self.authors)
        while neg_author == author:
            neg_author = random.choice(self.authors)
        negative_rec = random.choice(self.by_author[neg_author])

        return {
            "anchor": self._load(anchor_rec.page_path),
            "positive": self._load(positive_rec.page_path),
            "negative": self._load(negative_rec.page_path),
            "anchor_author": anchor_rec.author_id,
            "positive_author": positive_rec.author_id,
            "negative_author": negative_rec.author_id,
            "anchor_book": anchor_rec.book_id,
            "positive_book": positive_rec.book_id,
            "negative_book": negative_rec.book_id,
            "anchor_path": anchor_rec.page_path,
            "positive_path": positive_rec.page_path,
            "negative_path": negative_rec.page_path,
        }
