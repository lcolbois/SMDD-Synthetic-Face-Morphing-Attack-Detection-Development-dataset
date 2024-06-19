import os
import copy
import pandas as pd
import csv
import logging
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from utils import performances_compute
from backbones import mixnet_s
from mada.config import get_hydra_config_path
from mada.utils import fix_randomness
import hydra
from omegaconf import DictConfig
from mada.config.protocols import (
    list_preprocessed_samples,
    list_subsets_for_gathered_set,
    folds_to_group,
)
from mada.utils import get_global_seed


device = "cuda" if torch.cuda.is_available() else "cpu"

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]
INPUT_SIZE = 224
EarlyStopPatience = 20


class FaceDataset(Dataset):
    def __init__(self, samples_df, is_train=True):
        self.data = samples_df[["path", "attack"]]
        self.is_train = is_train
        self.train_transform = transforms.Compose(
            [
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=PRE__MEAN, std=PRE__STD),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
                transforms.ToTensor(),
                transforms.Normalize(mean=PRE__MEAN, std=PRE__STD),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        image = Image.open(item["path"]).convert("RGB")
        label = 0 if item["attack"] == "bonafide" else 1

        try:
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.test_transform(image)
        except ValueError:
            raise ValueError(f"Error with image {item['path']}")

        return image, label

    def get_normed_weights(self):
        # compute loss weights to improve the unbalance between data

        bonafide_num = (self.data["attack"] == "bonafide").sum()
        attack_num = (self.data["attack"] != "bonafide").sum()
        n_samples = [attack_num, bonafide_num]
        normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
        return torch.FloatTensor(normed_weights)


def train_fn(model, data_loader, data_size, optimizer, criterion):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / data_size
    epoch_acc = running_corrects.double() / data_size

    print("{} Loss: {:.4f} Acc: {:.4f}".format("Train", epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def eval_fn(model, data_loader, data_size, criterion):
    model.eval()

    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0

        prediction_scores, gt_labels = [], []
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            probs = F.softmax(outputs, dim=1)
            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i][1].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / data_size
        epoch_acc = running_corrects.double() / data_size
        _, eer_value, _ = performances_compute(
            prediction_scores, gt_labels, verbose=False
        )

        print(
            "{} Loss: {:.4f} Acc: {:.4f} EER: {:.4f}".format(
                "Val", epoch_loss, epoch_acc, eer_value
            )
        )

    return epoch_loss, epoch_acc, eer_value


def run_training(
    model,
    model_path,
    logging_path,
    normedWeights,
    num_epochs,
    dataloaders,
    dataset_sizes,
):
    model = model.to(device)
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=normedWeights).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6
    )

    logging.basicConfig(filename=logging_path, level=logging.INFO)

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_eer = 100
    epochs_no_improve = 0
    early_stop = False

    tb_writer = SummaryWriter(log_dir=model_path / "tb_logs")

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)
        tb_writer.add_scalar("epoch", epoch, epoch)
        # Each epoch has a training and validation phase
        train_loss, train_acc = train_fn(
            model, dataloaders["train"], dataset_sizes["train"], optimizer, criterion
        )
        val_loss, val_acc, val_eer_values = eval_fn(
            model, dataloaders["val"], dataset_sizes["val"], criterion
        )
        logging.info(
            "train loss: {}, train acc: {}, val loss: {}, val acc: {}, val eer: {}".format(
                train_loss, train_acc, val_loss, val_acc, val_eer_values
            )
        )

        tb_writer.add_scalar("loss/train", train_loss, epoch)
        tb_writer.add_scalar("acc/train", train_acc, epoch)
        tb_writer.add_scalar("loss/val", val_loss, epoch)
        tb_writer.add_scalar("acc/val", val_acc, epoch)
        tb_writer.add_scalar("eer/val", val_eer_values, epoch)

        # deep copy the model
        if val_eer_values <= lowest_eer:
            lowest_eer = val_eer_values
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_path / "best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        torch.save(model, model_path / "last.pth")

        if epochs_no_improve == EarlyStopPatience or epoch >= num_epochs:
            early_stop = True
        else:
            continue

        if early_stop:
            print("Trian process Stopped")
            print("epoch: {}".format(epoch))
            break

    print("Lowest EER: {:4f}".format(lowest_eer))
    logging.info("Lowest EER: {:4f}".format(lowest_eer))
    logging.info(f"saved model path: {model_path}")


def run_test(test_loader, model, model_path, batch_size=64):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction_scores, gt_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # probs = F.softmax(outputs, dim=1)

            for i in range(outputs.shape[0]):
                prediction_scores.append(float(outputs[i][1].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))

        # What the hell ? Compute stats in evaluation mode ???
        # std_value = np.std(prediction_scores)
        # mean_value = np.mean(prediction_scores)
        # prediction_scores = [ (float(i) - mean_value) /(std_value) for i in prediction_scores]
        _, eer_value, _ = performances_compute(
            prediction_scores, gt_labels, verbose=False
        )
        print(f"Test EER value: {eer_value*100}")

    return prediction_scores


def write_scores(test_csv, prediction_scores, output_path):
    save_data = []
    dataframe = pd.read_csv(test_csv)
    for idx in range(len(dataframe)):
        image_path = dataframe.iloc[idx, 0]
        label = dataframe.iloc[idx, 1]
        label = label.replace(" ", "")
        save_data.append(
            {
                "image_path": image_path,
                "label": label,
                "prediction_score": prediction_scores[idx],
            }
        )

    with open(output_path, mode="w") as csv_file:
        fieldnames = ["image_path", "label", "prediction_score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in save_data:
            writer.writerow(data)

    print(f"Saving prediction scores in {output_path}.")


@hydra.main(
    version_base=None,
    config_path=get_hydra_config_path(),
    config_name="mixfacenet_mad_training",
)
def main(cfg: DictConfig):
    fix_randomness(cfg.seed)
    df = pd.concat(
        [
            list_preprocessed_samples(**subset).assign(**subset)
            for subset in list_subsets_for_gathered_set(cfg.train_set)
        ]
    )
    df["group"] = folds_to_group(df["fold"])

    train_dataset = FaceDataset(
        df.query('group == "train"').sample(frac=1, random_state=get_global_seed()),
        is_train=True,
    )
    test_dataset = FaceDataset(
        df.query('group == "test"').sample(frac=1, random_state=get_global_seed()),
        is_train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = mixnet_s(embedding_size=128, width_scale=1.0, gdw_size=1024, shuffle=False)

    dataloaders = {"train": train_loader, "val": test_loader}
    dataset_sizes = {"train": len(train_dataset), "val": len(test_dataset)}
    print("train and test length:", len(train_dataset), len(test_dataset))

    logging_path = os.path.join(cfg.output_dir, "train_info.log")
    run_training(
        model=model,
        model_path=cfg.output_dir,
        logging_path=logging_path,
        normedWeights=train_dataset.get_normed_weights(),
        num_epochs=cfg.max_epochs,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
    )


if __name__ == "__main__":
    main()
