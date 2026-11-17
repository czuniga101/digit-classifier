"""Train a small feedforward network on MNIST, save it, reload it, and test it."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Settings ---
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MODEL_PATH = "model.pt"

# Use Apple GPU (MPS) or CUDA if available, otherwise CPU
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# --- Model: 784 -> 128 -> 64 -> 10 with ReLU between layers ---
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),             # 28x28 image -> 784 vector
            nn.Linear(28 * 28, 128),  # hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),       # hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 10),        # one output (logit) per digit
        )

    def forward(self, x):
        return self.net(x)


def main():
    # --- Data: convert to tensors and normalise with MNIST's mean/std ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    # --- Model, loss and optimiser ---
    model = DigitNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()                     # clear old gradients
            loss = loss_fn(model(images), labels)     # forward pass + loss
            loss.backward()                           # backpropagate
            optimiser.step()                          # update weights
            total_loss += loss.item() * images.size(0)

        # Evaluate accuracy on the test set after each epoch
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"loss {total_loss / len(train_set):.4f}  "
              f"test acc {correct / len(test_set):.2%}")

    # --- Save the trained weights ---
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    # --- Load into a fresh model and predict a handful of test images ---
    loaded = DigitNet().to(device)
    loaded.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    loaded.eval()

    print("\nSample predictions from the reloaded model:")
    with torch.no_grad():
        for i in range(10):
            image, label = test_set[i]
            pred = loaded(image.unsqueeze(0).to(device)).argmax(1).item()
            mark = "✓" if pred == label else "✗"
            print(f"  image {i}: predicted {pred}  true {label}  {mark}")


if __name__ == "__main__":
    main()
