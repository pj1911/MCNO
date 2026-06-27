"""Training script for MCNO on 1D PDE benchmarks."""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from timeit import default_timer
from scipy.io import loadmat

from mcno.model import Net1d
from mcno.utils import LpLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MCNO on 1D PDE benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", type=str, default="data/kdv_train_test.mat",
                        help="Path to the .mat dataset file")
    parser.add_argument("--ntrain", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--ntest", type=int, default=100,
                        help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for Adam optimizer")
    parser.add_argument("--step_size", type=int, default=100,
                        help="LR scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="LR scheduler decay factor")
    parser.add_argument("--width", type=int, default=64,
                        help="Model width (hidden channels)")
    parser.add_argument("--num_samples", type=int, default=150,
                        help="Number of MC sample points")
    parser.add_argument("--sub", type=int, default=32,
                        help="Subsampling rate for input resolution")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save trained model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    raw = loadmat(args.data_path)
    x_data = raw["input"].astype(np.float32)
    y_data = raw["output"].astype(np.float32)

    s = x_data.shape[1] // args.sub
    x_train = torch.from_numpy(x_data[: args.ntrain, :: args.sub])
    y_train = torch.from_numpy(y_data[: args.ntrain, :: args.sub])
    x_test = torch.from_numpy(x_data[-args.ntest :, :: args.sub])
    y_test = torch.from_numpy(y_data[-args.ntest :, :: args.sub])

    grid = torch.linspace(0, 1, s).reshape(1, s, 1)
    x_train = torch.cat([x_train.reshape(args.ntrain, s, 1), grid.repeat(args.ntrain, 1, 1)], dim=2)
    x_test = torch.cat([x_test.reshape(args.ntest, s, 1), grid.repeat(args.ntest, 1, 1)], dim=2)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size, shuffle=False,
    )

    # Model
    model = Net1d(args.width, s, args.num_samples).to(device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = LpLoss(size_average=False)

    # Training
    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch = x.size(0)

            optimizer.zero_grad()
            out = model(x)
            mse = F.mse_loss(out, y, reduction="mean")
            l2 = loss_fn(out.view(batch, -1), y.view(batch, -1))
            l2.backward()
            optimizer.step()

            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                batch = x.size(0)
                out = model(x)
                test_l2 += loss_fn(out.view(batch, -1), y.view(batch, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= args.ntrain
        test_l2 /= args.ntest
        t2 = default_timer()

        print(f"Epoch {ep:4d} | Time {t2-t1:.2f}s | Train MSE {train_mse:.6f} | "
              f"Train L2 {train_l2:.6f} | Test L2 {test_l2:.6f}")

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
