# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains


from modelFuncs import *
from ImageFuncs import load_data

cli_val = get_cli_args()
device = torch.device("cuda" if cli_val.gpu and torch.cuda.is_available() else "cpu")

model, fc, input_neuron = get_model(cli_val.arch)
model, optimizer = build_model_optim(
    model, input_neuron, fc, cli_val.hidden_units, cli_val.learning_rate
)

model.to(device)
# load data
train_dataloader, valid_dataloader, train_dataset = load_data(cli_val.data_directory)
# Train model
train_model(
    model, cli_val.epochs, optimizer, train_dataloader, valid_dataloader, device=device
)
# save_checkpoint
save_model(
    model,
    optimizer,
    train_dataset,
    cli_val.save_dir,
    cli_val.arch,
    cli_val.learning_rate,
    cli_val.hidden_units,
)
