# src/daneel/dream/gan_dream.py
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ----------------- DCGAN-STYLE GENERATOR -----------------
class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int = 32, nc: int = 1):
        """
        nz  = size of latent vector z (will be shaped into (nz,1,1) for ConvTranspose2d)
        ngf = feature maps in generator
        nc  = number of channels (1 for lightcurves)
        """
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) x 1 x 1 -> (ngf*4) x 11 x 11
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=11, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4) x 11 x 11 -> (ngf*2) x 22 x 22
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2) x 22 x 22 -> nc x 44 x 44
            nn.ConvTranspose2d(ngf * 2, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # expect z shaped (batch, nz, 1, 1)
        return self.main(z)


# ----------------- DCGAN-STYLE DISCRIMINATOR -----------------
class Discriminator(nn.Module):
    def __init__(self, ndf: int = 32, nc: int = 1):
        """
        ndf = feature maps in discriminator
        nc  = number of channels (1 for lightcurves)
        """
        super().__init__()
        self.main = nn.Sequential(
            # input: nc x 44 x 44 -> (ndf) x 22 x 22
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 22 x 22 -> (ndf*2) x 11 x 11
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 11 x 11 -> 1 x 1 x 1
            nn.Conv2d(ndf * 2, 1, kernel_size=11, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        return out.view(-1)


# ----------------- GAN TRAINER FOR TASK I -----------------
class GANDreamer:
    """
    Train a DCGAN-style network on light curves and 'dream' a new transit.
    """

    def __init__(self, params: dict):
        dream_params = params.get("dream", {})

        # dataset and preprocessing
        self.dataset_path = Path(dream_params.get("dataset_path", "tess_data.csv"))
        self.label_column: Optional[str] = dream_params.get("label_column", None)

        # hyperparameters
        self.batch_size = int(dream_params.get("batch_size", 128))
        self.nz = int(dream_params.get("z_dim", 100))
        self.ngf = int(dream_params.get("ngf", 32))
        self.ndf = int(dream_params.get("ndf", 32))
        self.num_epochs = int(dream_params.get("epochs", 100))
        self.lr = float(dream_params.get("learning_rate", 0.0002))
        self.beta1 = float(dream_params.get("beta1", 0.5))

        # model files / outputs
        self.model_dir = Path(dream_params.get("model_dir", "models/gan"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gen_path = self.model_dir / "netG.pt"
        self.disc_path = self.model_dir / "netD.pt"
        self.sample_output = Path(dream_params.get("sample_output", "assignment2_taskI_dream.png"))

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.netG = Generator(self.nz, self.ngf, nc=1).to(self.device)
        self.netD = Discriminator(self.ndf, nc=1).to(self.device)

        # loss + optimizers
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    # ---------- DATA: read CSV, coerce numeric, cut to 1936, reshape ----------
    def _load_dataset(self) -> DataLoader:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)

        # drop label column if present
        if self.label_column is not None and self.label_column in df.columns:
            df = df.drop(columns=[self.label_column])

        # coerce all columns to numeric (non-convertible -> NaN)
        df_numeric = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

        # drop non-numeric columns and incomplete rows
        df_numeric = df_numeric.dropna(axis=1, how="all")   # remove columns all-NaN
        df_numeric = df_numeric.dropna(axis=0, how="any")   # remove rows with any NaN

        if df_numeric.shape[0] == 0:
            raise ValueError("No numeric rows found in the dataset after coercion. Check CSV / label_column.")

        X = df_numeric.values.astype(np.float32)

        # require at least 1936 points (assignment hint)
        if X.shape[1] < 1936:
            raise ValueError(f"Expected at least 1936 flux points, got {X.shape[1]} (columns).")

        X = X[:, :1936]

        # normalize to [-1, 1]
        max_abs = np.max(np.abs(X))
        if max_abs == 0:
            max_abs = 1.0
        X = X / max_abs

        # reshape to (N, 1, 44, 44)
        X = X.reshape(-1, 1, 44, 44)

        tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # ---------- TRAINING LOOP ----------
    def train(self) -> None:
        dataloader = self._load_dataset()

        real_label = 1.0
        fake_label = 0.0

        print("Starting GAN training loop...")
        for epoch in range(self.num_epochs):
            for i, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                b_size = real_batch.size(0)

                # ---- (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
                self.netD.zero_grad()

                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_batch)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(fake_label)
                output_fake = self.netD(fake.detach())
                errD_fake = self.criterion(output_fake, label)
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()

                errD = errD_real + errD_fake
                self.optimizerD.step()

                # ---- (2) Update G: maximize log(D(G(z)))
                self.netG.zero_grad()
                label.fill_(real_label)
                output_fake_for_G = self.netD(fake)
                errG = self.criterion(output_fake_for_G, label)
                errG.backward()
                D_G_z2 = output_fake_for_G.mean().item()
                self.optimizerG.step()

            print(
                f"[Epoch {epoch+1}/{self.num_epochs}] "
                f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
            )

        # save models and produce sample
        torch.save(self.netG.state_dict(), self.gen_path)
        torch.save(self.netD.state_dict(), self.disc_path)
        self.dream_and_save()

    # ---------- GENERATE & SAVE ONE DREAM ----------
    def dream_and_save(self) -> None:
        self.netG.eval()
        with torch.no_grad():
            noise = torch.randn(1, self.nz, 1, 1, device=self.device)
            fake_img = self.netG(noise).cpu().numpy()[0, 0]  # (44,44)

        curve = fake_img.reshape(-1)  # length 1936
        x = np.arange(len(curve))

        plt.figure(figsize=(6, 4))
        plt.plot(x, curve)
        plt.xlabel("Time index")
        plt.ylabel("Normalised flux")
        plt.title("Dreamed exoplanetary transit (GAN)")
        plt.tight_layout()
        self.sample_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.sample_output)
        plt.close()

        print(f"[GAN] Dreamed light curve saved to: {self.sample_output}")

    # ---------- PUBLIC ENTRYPOINT ----------
    def run(self) -> None:
        self.train()

