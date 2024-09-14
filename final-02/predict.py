import torch
from predictFuncs import get_cli_args, load_model, predict

cli_val = get_cli_args()

device = torch.device("cuda" if cli_val.gpu and torch.cuda.is_available() else "cpu")

model, optimizer = load_model(cli_val.checkpoint, device)
model.to(device)
probs, classes = predict(
    cli_val.image_path, device, model, cli_val.top_k, cli_val.category_names
)

print("-" * 75)
print("Probability           -        Classes")
for p, c in zip(probs, classes):
    print(f"{p:.4f}           -          {c}")
