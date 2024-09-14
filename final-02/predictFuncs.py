import argparse
import torch
import json
from modelFuncs import get_model, build_model_optim
from ImageFuncs import process_image


def get_cli_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window
    * Return top K most likely classes: python predict.py input checkpoint --top_k 3
    * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    * Use GPU for inference: python predict.py input checkpoint --gpu
    """
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "image_path", type=str, help="File pathe of the Image to classify"
    )
    my_parser.add_argument("checkpoint", type=str, help="Checkpoint file path")
    my_parser.add_argument(
        "--top_k", type=int, default=1, help="Count of most likely classes"
    )
    my_parser.add_argument(
        "--category_names",
        type=str,
        help="Index to categories json file path",
        default="cat_to_name.json",
    )
    my_parser.add_argument("--gpu", action="store_true", help="Allow GPU for inference")

    return my_parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, fc, input_neuron = get_model(checkpoint["model_name"])
    model, optimizer = build_model_optim(
        model, input_neuron, fc, checkpoint["hidden_units"], checkpoint["lr"]
    )

    model.load_state_dict(checkpoint["model_vals"])
    model.class_to_idx = checkpoint["cls_to_idx"]

    optimizer.load_state_dict(checkpoint["optim_vals"])
    print("-" * 75)
    print("Successfully Loaded Model")
    print("-" * 75)
    return model, optimizer


def predict(image_path, device, model, topk, category_names):
    """Predict the class (or classes) of an image using a trained deep learning model."""

    with open(category_names, "r") as file:
        category_names = json.load(file)

    transformed_img = process_image(image_path)
    output = model(transformed_img.unsqueeze(0).float().to(device))
    predictions = torch.softmax(output, dim=1)
    probs, classes = predictions.topk(topk, dim=1)

    class_dict = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))

    classes = [class_dict[cls] for cls in classes[0].tolist()]
    classes = [category_names[cls] for cls in classes]
    return probs.tolist()[0], classes
