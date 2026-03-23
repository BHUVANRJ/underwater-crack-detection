"""
=============================================================
  BIJECTIVE GAN - UNDERWATER CRACK SYNTHETIC DATA GENERATION
  Optimized for: RTX 5060 Laptop GPU (8GB VRAM)
=============================================================
WHAT THIS DOES:
  1. Trains a Bijective GAN on your 10,230 crack images
  2. Generates 10,000 new synthetic crack images
  3. Saves them ready for YOLOv8 training

HOW TO RUN:

  Step 1 - Train the GAN (run from C:\crack_project):
    python scripts/bijective_gan.py --mode train --data_dir dataset/train_augmented/images --output_dir gan_output --epochs 100

  Step 2 - Generate new images:
    python scripts/bijective_gan.py --mode generate --output_dir gan_output --num_images 10000 --save_dir dataset/gan_generated/images

REQUIREMENTS:
  pip install torch torchvision tqdm Pillow numpy
=============================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path


# =============================================================
#  SECTION 1: CONFIGURATION
# =============================================================

class Config:
    IMAGE_SIZE    = 256
    CHANNELS      = 3
    LATENT_DIM    = 128
    FEATURES_G    = 64
    FEATURES_D    = 64
    FEATURES_E    = 64
    BATCH_SIZE    = 8          # Safe for 8GB VRAM
    LR_G          = 0.0002
    LR_D          = 0.0002
    BETA1         = 0.5
    BETA2         = 0.999
    LAMBDA_RECON  = 10.0       # Image reconstruction weight
    LAMBDA_LATENT = 0.5        # Latent reconstruction weight (bijective constraint)
    LAMBDA_KL     = 0.01       # KL divergence weight
    EPOCHS        = 100
    SAVE_EVERY    = 10
    SAMPLE_EVERY  = 5
    NUM_WORKERS   = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()


# =============================================================
#  SECTION 2: DATASET
# =============================================================

class CrackDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = [
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in extensions
        ]
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        print(f"  Dataset: {len(self.image_files)} images loaded")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        return self.transform(img)


# =============================================================
#  SECTION 3: NETWORK ARCHITECTURE
# =============================================================

class ResBlock(nn.Module):
    """Residual block for training stability."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class Generator(nn.Module):
    """
    z (128,) → crack image (3, 256, 256)
    Upsamples from 4x4 to 256x256 through 6 stages.
    """
    def __init__(self, latent_dim=128, features=64):
        super().__init__()
        self.init_size = 4
        f = features

        self.project = nn.Sequential(
            nn.Linear(latent_dim, f * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(f * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResBlock(out_ch),
            )

        self.blocks = nn.Sequential(
            up_block(f * 8, f * 8),   # 4   → 8
            up_block(f * 8, f * 4),   # 8   → 16
            up_block(f * 4, f * 4),   # 16  → 32
            up_block(f * 4, f * 2),   # 32  → 64
            up_block(f * 2, f),       # 64  → 128
            up_block(f,     f // 2),  # 128 → 256
            nn.Conv2d(f // 2, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.project(z)
        x = x.view(z.size(0), cfg.FEATURES_G * 8, self.init_size, self.init_size)
        return self.blocks(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator — judges local image patches.
    Better at detecting texture artifacts in crack images.
    (3, 256, 256) → (1, 8, 8) patch scores
    """
    def __init__(self, features=64):
        super().__init__()
        f = features

        def down_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.1))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            down_block(3,     f,     normalize=False),
            down_block(f,     f * 2),
            down_block(f * 2, f * 4),
            down_block(f * 4, f * 8),
            down_block(f * 8, f * 8),
            nn.Conv2d(f * 8, 1, 3, 1, 1),
        )

    def forward(self, img):
        return self.model(img)


class Encoder(nn.Module):
    """
    THE BIJECTIVE COMPONENT — maps images back to latent space.

    WHY THIS MAKES IT BIJECTIVE:
    ─────────────────────────────
    Standard GAN: multiple z values → same image (mode collapse)
    Bijective GAN: enforces one-to-one mapping via:
      Forward:  z → G(z) → image
      Inverse:  image → E(image) → z_recovered

    Latent Reconstruction Loss: ||z - E(G(z))||
    forces every unique z to produce a unique image.

    Uses VAE-style encoder with reparameterization trick
    for smooth, continuous latent space navigation.
    """
    def __init__(self, latent_dim=128, features=64):
        super().__init__()
        f = features

        def down_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.conv = nn.Sequential(
            down_block(3,     f,     normalize=False),
            down_block(f,     f * 2),
            down_block(f * 2, f * 4),
            down_block(f * 4, f * 8),
            down_block(f * 8, f * 8),
            down_block(f * 8, f * 8),
        )

        flat = f * 8 * 4 * 4
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(-10, 10)
        z      = self.reparameterize(mu, logvar)
        return z, mu, logvar


# =============================================================
#  SECTION 4: TRAINING
# =============================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(data_dir, output_dir, epochs):
    output_path    = Path(output_dir)
    checkpoint_dir = output_path / "checkpoints"
    sample_dir     = output_path / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BIJECTIVE GAN — UNDERWATER CRACK DETECTION")
    print("=" * 60)
    print(f"  Device:     {cfg.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {cfg.BATCH_SIZE}")
    print(f"  Image size: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"  Latent dim: {cfg.LATENT_DIM}")
    print("=" * 60)

    dataset    = CrackDataset(data_dir, cfg.IMAGE_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.DEVICE.type == 'cuda',
        drop_last=True
    )

    G = Generator(cfg.LATENT_DIM, cfg.FEATURES_G).to(cfg.DEVICE)
    D = Discriminator(cfg.FEATURES_D).to(cfg.DEVICE)
    E = Encoder(cfg.LATENT_DIM, cfg.FEATURES_E).to(cfg.DEVICE)

    G.apply(weights_init)
    D.apply(weights_init)
    E.apply(weights_init)

    # G and E optimized together
    opt_GE = optim.Adam(list(G.parameters()) + list(E.parameters()),
                        lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    opt_D  = optim.Adam(D.parameters(),
                        lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    def lr_lambda(epoch):
        decay_start = epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / max(1, epochs - decay_start)

    sched_GE = optim.lr_scheduler.LambdaLR(opt_GE, lr_lambda)
    sched_D  = optim.lr_scheduler.LambdaLR(opt_D,  lr_lambda)

    adv_loss   = nn.MSELoss()
    recon_loss = nn.L1Loss()

    fixed_z = torch.randn(16, cfg.LATENT_DIM).to(cfg.DEVICE)
    history = {'d_loss': [], 'g_loss': [], 'recon': [], 'latent': []}

    print(f"\n  Training started! Samples saved every {cfg.SAMPLE_EVERY} epochs.")
    print(f"  Checkpoints saved every {cfg.SAVE_EVERY} epochs.\n")

    for epoch in range(1, epochs + 1):
        G.train(); D.train(); E.train()
        epoch_d = epoch_g = epoch_r = epoch_l = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}/{epochs}", leave=True)

        for real_imgs in pbar:
            real_imgs  = real_imgs.to(cfg.DEVICE)
            batch_size = real_imgs.size(0)

            with torch.no_grad():
                patch_shape = D(real_imgs).shape
            real_labels = torch.ones(patch_shape).to(cfg.DEVICE)
            fake_labels = torch.zeros(patch_shape).to(cfg.DEVICE)

            # ── Discriminator ──────────────────────────────
            opt_D.zero_grad()
            d_real = adv_loss(D(real_imgs), real_labels)
            z      = torch.randn(batch_size, cfg.LATENT_DIM).to(cfg.DEVICE)
            d_fake = adv_loss(D(G(z).detach()), fake_labels)
            d_loss = (d_real + d_fake) * 0.5
            d_loss.backward()
            opt_D.step()

            # ── Generator + Encoder ────────────────────────
            opt_GE.zero_grad()

            # 1. Adversarial
            z     = torch.randn(batch_size, cfg.LATENT_DIM).to(cfg.DEVICE)
            g_adv = adv_loss(D(G(z)), real_labels)

            # 2. Reconstruction: G(E(x)) ≈ x
            z_enc, mu, logvar = E(real_imgs)
            g_recon = recon_loss(G(z_enc), real_imgs)

            # 3. Latent reconstruction (BIJECTIVE): E(G(z)) ≈ z
            z2         = torch.randn(batch_size, cfg.LATENT_DIM).to(cfg.DEVICE)
            z2_rec,_,_ = E(G(z2))
            g_latent   = recon_loss(z2_rec, z2)

            # 4. KL divergence
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            g_loss = (g_adv
                      + cfg.LAMBDA_RECON  * g_recon
                      + cfg.LAMBDA_LATENT * g_latent
                      + cfg.LAMBDA_KL    * kl)

            g_loss.backward()
            nn.utils.clip_grad_norm_(
                list(G.parameters()) + list(E.parameters()), max_norm=1.0
            )
            opt_GE.step()

            epoch_d += d_loss.item()
            epoch_g += g_adv.item()
            epoch_r += g_recon.item()
            epoch_l += g_latent.item()
            n_batches += 1

            pbar.set_postfix({
                'D': f'{d_loss.item():.3f}',
                'G': f'{g_adv.item():.3f}',
                'Recon': f'{g_recon.item():.3f}',
                'Latent': f'{g_latent.item():.3f}',
            })

        sched_GE.step()
        sched_D.step()

        avg_d = epoch_d / n_batches
        avg_g = epoch_g / n_batches
        avg_r = epoch_r / n_batches
        avg_l = epoch_l / n_batches

        history['d_loss'].append(avg_d)
        history['g_loss'].append(avg_g)
        history['recon'].append(avg_r)
        history['latent'].append(avg_l)

        print(f"  Epoch {epoch:03d} | D: {avg_d:.4f} | G: {avg_g:.4f} | "
              f"Recon: {avg_r:.4f} | Latent: {avg_l:.4f}")

        if epoch % cfg.SAMPLE_EVERY == 0:
            G.eval()
            with torch.no_grad():
                samples = (G(fixed_z) + 1) / 2
                save_image(make_grid(samples, nrow=4),
                           sample_dir / f"epoch_{epoch:04d}.png")
            print(f"  → Sample saved: gan_output/samples/epoch_{epoch:04d}.png")

        if epoch % cfg.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'G_state': G.state_dict(),
                'D_state': D.state_dict(),
                'E_state': E.state_dict(),
                'opt_GE': opt_GE.state_dict(),
                'opt_D':  opt_D.state_dict(),
                'history': history,
            }, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth")
            print(f"  → Checkpoint saved: checkpoint_epoch_{epoch:04d}.pth")

    torch.save({
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'E_state': E.state_dict(),
        'config': {
            'latent_dim': cfg.LATENT_DIM,
            'image_size': cfg.IMAGE_SIZE,
            'features_g': cfg.FEATURES_G,
        }
    }, output_path / "bijective_gan_final.pth")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Model: {output_dir}/bijective_gan_final.pth")
    print(f"  Samples: {output_dir}/samples/")
    print("\n  Now generate synthetic images:")
    print(f"  python scripts/bijective_gan.py --mode generate --output_dir {output_dir} --num_images 10000")
    print("=" * 60)


# =============================================================
#  SECTION 5: IMAGE GENERATION
# =============================================================

def generate(output_dir, num_images, save_dir):
    output_path = Path(output_dir)
    save_path   = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "bijective_gan_final.pth"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run training first: --mode train")
        return

    print("\n" + "=" * 60)
    print("  GENERATING SYNTHETIC CRACK IMAGES")
    print("=" * 60)
    print(f"  Generating: {num_images} images")
    print(f"  Saving to:  {save_dir}")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    model_cfg  = checkpoint.get('config', {})

    latent_dim = model_cfg.get('latent_dim', cfg.LATENT_DIM)
    features_g = model_cfg.get('features_g', cfg.FEATURES_G)

    G = Generator(latent_dim, features_g).to(cfg.DEVICE)
    G.load_state_dict(checkpoint['G_state'])
    G.eval()

    generated  = 0
    batch_size = 32

    with torch.no_grad():
        with tqdm(total=num_images, desc="Generating") as pbar:
            while generated < num_images:
                current_batch = min(batch_size, num_images - generated)
                z    = torch.randn(current_batch, latent_dim).to(cfg.DEVICE)
                imgs = ((G(z) + 1) / 2).clamp(0, 1)

                for i in range(current_batch):
                    img_np  = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                    Image.fromarray(img_np).save(
                        save_path / f"synthetic_crack_{generated + i:06d}.png"
                    )

                generated += current_batch
                pbar.update(current_batch)

    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE!")
    print(f"  {generated} synthetic images saved to {save_dir}")
    print("\n  Next step: Train YOLOv8 on your full dataset!")
    print("=" * 60)


# =============================================================
#  SECTION 6: CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bijective GAN for Crack Image Generation")
    parser.add_argument("--mode", required=True, choices=["train", "generate"])
    parser.add_argument("--data_dir",   default="dataset/train_augmented/images")
    parser.add_argument("--output_dir", default="gan_output")
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--save_dir",   default="dataset/gan_generated/images")
    args = parser.parse_args()

    cfg.BATCH_SIZE = args.batch_size

    if args.mode == "train":
        train(args.data_dir, args.output_dir, args.epochs)
    else:
        generate(args.output_dir, args.num_images, args.save_dir)
