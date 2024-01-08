import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # self.linear = nn.Linear(1, 1)

        # this should be randomly initialized
        self.w = nn.Parameter(torch.tensor(4.5))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x + self.b


def main() -> None:
    w_hat: float = 3.0
    b_hat: float = 2.0

    num_samples: int = 500
    num_epochs: int = 100

    x: torch.Tensor = torch.linspace(-2, 2, num_samples)

    def f(x: torch.Tensor) -> torch.Tensor:
        return w_hat * x + b_hat

    noise = torch.randn(num_samples)

    # some synthesized data augmented with gaussian noise
    y = f(x) + noise

    model: Model = Model()

    def loss_fn(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)

    with torch.no_grad():
        plt.plot(x, y, ".", label="Data")
        plt.plot(x, f(x), label="Ground truth")
        plt.plot(x, model(x), label="Predictions")
        plt.legend()

        print("Current loss: %1.6f" % loss_fn(y, model(x)).numpy())

    def training_loop(model: nn.Module, x: torch.Tensor, y: torch.Tensor, num_epochs: int) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            loss.backward()
            optimizer.step()

            print("Epoch %d, loss %1.6f" % (epoch, loss.item()))

    # usually the dataset is split into batches but in our case we can treat the entire dataset
    # as a single batch.
    training_loop(model, x, y, num_epochs)

    with torch.no_grad():
        plt.plot(x, model(x), label="Trained model predictions")
        plt.legend()

        print("Current loss: %1.6f" % loss_fn(y, model(x)).numpy())


if __name__ == "__main__":
    main()
