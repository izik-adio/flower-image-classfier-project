import os
import torch
from torch import nn, optim
import argparse
from torchvision.models import vgg13, resnet34, efficientnet_b0


def get_cli_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    * Choose architecture: python train.py data_dir --arch "vgg13"
    * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 960 --epochs 20
    * Use GPU for training: python train.py data_dir --gpu
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("data_directory", type=str, help="Dataset Folder")
    my_parser.add_argument(
        "--save_dir", type=str, help="Checkpoint Folder", default="Checkpoints"
    )
    my_parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet",
        choices=["effieicentnet", "vgg", "resnet"],
        help="Pretrained Model Architecture",
    )
    my_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Hyperparameter: Learning Rate",
    )
    my_parser.add_argument(
        "--hidden_units",
        type=int,
        default=960,
        help="Hyperparameter: Count of Hidden Neurons Per Layer",
    )
    my_parser.add_argument(
        "--epochs", type=int, default=20, help="Hyperparameter: Number Of Epochs"
    )
    my_parser.add_argument(
        "--gpu", action="store_true", help="Allow GPU for model Training"
    )

    return my_parser.parse_args()


def get_model(model_name):
    match model_name:
        case "vgg":
            model = vgg13(weights="DEFAULT")
            fc = "classifier"
            input_neuron = 25088
        case "resnet":
            model = resnet34(weights="DEFAULT")
            fc = "fc"
            input_neuron = 512
        case "efficientnet":
            model = efficientnet_b0(weights="DEFAULT")
            fc = "classifier"
            input_neuron = 1280
    return (model, fc, input_neuron)


def build_model_optim(model, input_neuron, fc, hidden_units, learning_rate):

    for param in model.parameters():
        param.requires_grad = False

    if fc == "classifier":
        model.classifier = Fclassfier(input_neuron, hidden_units)
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    else:
        model.fc = Fclassfier(input_neuron, hidden_units)
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    print("-" * 75)
    print("Successfully Built Model")
    print("-" * 75)
    return model, optimizer


class Fclassfier(nn.Module):
    def __init__(self, in_neuron, h__neuron, out_neuron=102):
        super(Fclassfier, self).__init__()

        self.hidden = nn.ModuleList()
        neurons = [in_neuron, h__neuron, 640]
        for x, y in zip(neurons, neurons[1:]):
            layer = nn.Linear(x, y)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            self.hidden.append(layer)

        self.hidden.append(nn.Linear(neurons[-1], out_neuron))
        self.relu = nn.ReLU()
        self.drops = [nn.Dropout(p) for p in [0.3, 0.4]]

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.hidden[:-1]):
            out = layer(out)
            out = self.relu(self.drops[i](out))
        out = self.hidden[-1](out)

        return out


def train_model(model, num_epochs, optimizer, train_loader, valid_loader, device):
    criterion = nn.CrossEntropyLoss()
    lr_steper = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)

    print("Started Training Model")
    print("-" * 75)
    for epoch in range(num_epochs):
        model.train()
        tr_losses = []
        va_losses = []
        accuracies = []

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            tr_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (i % 25) == 0:
                model.eval()

                with torch.no_grad():
                    for images, labels in valid_loader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        va_losses.append(loss.item())
                        _, prediction = torch.max(output, dim=1)
                        equal = (prediction == labels).float()
                        accuracies.append(equal.mean().item())

                model.train()

                mean_t_loss = torch.tensor(tr_losses).mean().item()
                mean_v_loss = torch.tensor(va_losses).mean().item()
                acc = torch.tensor(accuracies).mean().item()

                print(
                    f"Epoch: {epoch + 1}/{num_epochs} Step: {i} Training Loss: {mean_t_loss:.3f} Validation Loss: {mean_v_loss:.3f} Accuracy: {acc:.4f}%"
                )

        lr_steper.step()

    print("-" * 75)
    print("Successfully Trained Model")
    print("-" * 75)


def save_model(
    model, optimizer, train_dataset, save_dir, model_name, learning_rate, hidden_units
):
    checkpoint = {
        "model_vals": model.state_dict(),
        "model_name": model_name,
        "lr": learning_rate,
        "hidden_units": hidden_units,
        "optim_vals": optimizer.state_dict(),
        "cls_to_idx": train_dataset.class_to_idx,
    }

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = f"{save_dir}/{model_name}-checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print("Successfully Saved Model")
    print("-" * 75)
