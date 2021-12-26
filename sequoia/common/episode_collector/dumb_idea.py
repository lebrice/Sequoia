from torch import nn, Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader


class ContextLinear(nn.Linear):
    """ Probably very stupid idea. 
    
    IDEA: Figure out if its possible, somehow, to not just "feed" the input to the network,
    but also:
    idea #1: its own weights (?)
    idea #2: the weights of its neighbours (?).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        # self.context_weight = nn.Linear(in_features=out_features, out_features=out_features, bias=True)

    def forward(self, input: Tensor) -> Tensor:
        context: Tensor = sum(
            self.weight.roll([i, j], dims=[0, 1])
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            if not (i == 0 and j == 0)
        ) / 8

        weight = self.weight

        # Element-wise product of the 'neighbourhood' and of the neuron itself.
        contextualized_weight = weight * context.detach()

        output = input @ contextualized_weight.T + self.bias
        return output


import torch
from torch.utils.data import TensorDataset
from torch.optim import SGD


def main():
    a, b, c = 1, -2, 3
    dataset_size = 100
    seed = 123

    def f(x: Tensor) -> Tensor:
        return (a * x[:, 1] ** 2 + b * x[:, 1] + c).reshape([x.shape[0], 1])

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.rand([dataset_size, 2])
    y = f(x)

    dataset = TensorDataset(x, y)

    layers = [
        layer_type(2, 1)
        for layer_type in [nn.Linear, ContextLinear, nn.Linear, ContextLinear]
    ]
    import copy

    state_dict = copy.deepcopy(layers[0].state_dict())
    # starting_weight = torch.rand_like(layers[0].weight)
    # starting_bias = torch.rand_like(layers[0].bias)

    for layer in layers:
        print(f"Layer of type {layer}:")
        # NOTE: Not actually necessary.
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        layer.load_state_dict(state_dict)
        # layer.weight.data.copy_(starting_weight)
        # layer.bias.data.copy_(starting_bias)
        # print(layer.weight) # shouldn't change between the different layers!!

        optimizer = SGD(layer.parameters(), lr=1e-3)
        dataloader = DataLoader(dataset, batch_size=dataset_size)

        for epoch in range(10):
            y_pred = layer(x)

            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            if epoch == 0:
                print(f"Epoch {epoch}: loss: {loss.item()}")
            # print(epoch, loss)
            optimizer.step()

        print(f"Epoch {epoch}: loss: {loss.item()}")


if __name__ == "__main__":
    main()
