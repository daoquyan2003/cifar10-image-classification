import torch
from torch import nn
from torchvision import models

class CustomNet(nn.Module):
    """A custom network implementation based on pretrained neural network for computing predictions."""

    def __init__(self, 
                 num_classes: int = 10,
                 img_pretrained: str = None,
    ) -> None:
        """Initialize a `CustomNet` module.

        :param num_classes: The number of output features of the final linear layer.
        """
        super().__init__()

        if img_pretrained is None:
            self.name_pretrained = img_pretrained
        else:
            self.name_pretrained = 'efficientnet_b7'

        self.pretrained_model = models.get_model(name=self.name_pretrained, weights="DEFAULT")

        self.linear_layer = nn.Linear(1000, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.linear_layer(self.pretrained_model(x))
    

if __name__ == "__main__":
    _ = CustomNet()
