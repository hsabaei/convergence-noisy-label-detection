import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "computer_keyboard", "lamp", "lawn_mower", "leopard",
    "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm"
]

CLOSE_TO_CIFAR10 = {
    "bus", "pickup_truck", "streetcar", "tank", "tractor", "train",
    "bear", "beaver", "bee", "butterfly", "camel", "cattle", "fox", "leopard",
    "lion", "mouse", "possum", "rabbit", "raccoon", "skunk",
    "squirrel", "tiger", "wolf", "rocket",
}


def get_allowed_cifar100_classes():
    return [c for c in CIFAR100_FINE_LABELS if c not in CLOSE_TO_CIFAR10]


class CIFAR10WithIndex(Dataset):
    def __init__(self, root: str, train: bool, transform=None, download: bool = True):
        self.ds = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, idx


class MixedCIFAR10WithInjectedCIFAR100(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        download: bool = True,
        samples_per_c100_class: int = 10,
        seed: int = 66,
        allowed_c100_classes=None,
    ):
        super().__init__()
        self.transform = transform
        self.train = train

        self.c10 = datasets.CIFAR10(root=root, train=train, download=download)
        self.c100 = datasets.CIFAR100(root=root, train=train, download=download)

        rng = np.random.default_rng(seed)

        c10_data = self.c10.data
        c10_targets = np.asarray(self.c10.targets, dtype=np.int64)

        data_list = [img for img in c10_data]
        target_list = c10_targets.tolist()
        is_anomaly_list = [False] * len(c10_targets)
        source_list = ["cifar10"] * len(c10_targets)
        original_c100_label_list = [-1] * len(c10_targets)
        original_c100_name_list = [""] * len(c10_targets)

        if allowed_c100_classes is None:
            allowed_c100_classes = get_allowed_cifar100_classes()

        fine_to_idx = {name: i for i, name in enumerate(CIFAR100_FINE_LABELS)}
        c100_targets = np.asarray(self.c100.targets, dtype=np.int64)

        for fine_name in allowed_c100_classes:
            fine_id = fine_to_idx[fine_name]
            idx_all = np.where(c100_targets == fine_id)[0]

            if len(idx_all) < samples_per_c100_class:
                raise ValueError(
                    f"Class {fine_name} has only {len(idx_all)} samples, "
                    f"but requested {samples_per_c100_class}."
                )

            chosen = rng.choice(idx_all, size=samples_per_c100_class, replace=False)

            if samples_per_c100_class <= 10:
                fake_labels = np.arange(10)[:samples_per_c100_class].copy()
                rng.shuffle(fake_labels)
            else:
                fake_labels = np.tile(np.arange(10), int(np.ceil(samples_per_c100_class / 10)))
                fake_labels = fake_labels[:samples_per_c100_class]
                rng.shuffle(fake_labels)

            for j, fake_y in zip(chosen, fake_labels):
                img = self.c100.data[j]

                data_list.append(img)
                target_list.append(int(fake_y))
                is_anomaly_list.append(True)
                source_list.append("cifar100")
                original_c100_label_list.append(int(fine_id))
                original_c100_name_list.append(fine_name)

        self.data = np.stack(data_list, axis=0)
        self.targets = np.asarray(target_list, dtype=np.int64)
        self.is_anomaly = np.asarray(is_anomaly_list, dtype=bool)
        self.source = source_list
        self.original_c100_label = np.asarray(original_c100_label_list, dtype=np.int64)
        self.original_c100_name = original_c100_name_list

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        y = int(self.targets[idx])

        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, y, idx


class NoisyCIFAR10WithIndex(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        download: bool = True,
        noisy_frac: float = 0.2,
        seed: int = 66,
        num_classes: int = 10,
    ):
        super().__init__()

        self.ds = datasets.CIFAR10(root=root, train=train, transform=None, download=download)
        self.transform = transform
        self.data = self.ds.data
        self.true_targets = np.asarray(self.ds.targets, dtype=np.int64)
        self.targets = self.true_targets.copy()

        rng = np.random.default_rng(seed)
        N = len(self.targets)
        self.is_anomaly = np.zeros(N, dtype=bool)

        if train and noisy_frac > 0:
            num_noisy = int(noisy_frac * N)
            noisy_idx = rng.choice(N, size=num_noisy, replace=False)

            for i in noisy_idx:
                y = self.true_targets[i]
                choices = [c for c in range(num_classes) if c != y]
                self.targets[i] = rng.choice(choices)

            self.is_anomaly[noisy_idx] = True

        self.source = [
            "noisy_label" if self.is_anomaly[i] else "clean"
            for i in range(N)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        y_noisy = int(self.targets[idx])

        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, y_noisy, idx
