import torch
from torchvision.transforms import v2
import gradio as gr
import rootutils
import argparse

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.cifar10_module import CIFAR10LitModule

parser = argparse.ArgumentParser(description='Pass in the checkpoint path to load the model.')

parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file')

# Parse the command line arguments
args = parser.parse_args()

ckpt_path = args.ckpt_path

model = CIFAR10LitModule.load_from_checkpoint(checkpoint_path=ckpt_path).net.eval()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict(inp):
    transform = v2.Compose(
            [v2.ToImage(),
             v2.Resize((32, 32)),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {classes[i]: float(prediction[i]) for i in range(10)}

    return confidences

demo = gr.Interface(fn=predict,
             inputs=gr.Image(type='pil'),
             outputs=gr.Label(num_top_classes=3))

demo.launch(share=True)