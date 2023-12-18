import torch
from tqdm import tqdm
import numpy as np
import wandb
from metrics import SSIM_metric, FID_metric
from torch.utils.data import DataLoader
from dataset import GeneratedImagesDataset

@torch.no_grad()
def eval_epoch(data_loader, model, device):
    # compute SSIM metric
    total_ssim = []
    for real_images, _ in tqdm(data_loader):
        real_images = real_images.to(device)
        latent = torch.randn(len(real_images), model['generator'].latent_size, 1, 1, device=device)
        generated_images = model['generator'](latent)

        total_ssim.append(SSIM_metric(generated_images=generated_images, real_images=real_images))

    # compute FID metric
    generated_dataset = GeneratedImagesDataset(generated_images)
    generated_dataloader = DataLoader(generated_dataset, batch_size=data_loader.batch_size)
    fid = FID_metric(generated_dataloader=generated_dataloader, real_dataloader=data_loader, device=device)
    return np.mean(total_ssim),fid


def train_epoch(train_loader, model, optimizer, criterion, device):
    train_stats = {'loss_d': [], 'loss_g': [], 'real_score': [], 'fake_score': []}

    for real_images, _ in tqdm(train_loader):
        batch_size = real_images.shape[0]
        real_images = real_images.to(device)
        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # Train discriminator
        optimizer["discriminator"].zero_grad()
        real_preds = model["discriminator"](real_images)
        fake_images = model["generator"](torch.randn(batch_size, model["generator"].latent_size, 1, 1, device=device))
        fake_preds = model["discriminator"](fake_images.detach())

        loss_d = criterion["discriminator"](real_preds.view(real_targets.shape), real_targets) + criterion["discriminator"](fake_preds.view(fake_targets.shape), fake_targets)
        loss_d.backward()
        optimizer["discriminator"].step()

        # Train generator
        optimizer["generator"].zero_grad()
        gen_preds = model["discriminator"](fake_images)
        loss_g = criterion["generator"](gen_preds.view(real_targets.shape), real_targets)
        loss_g.backward()
        optimizer["generator"].step()

        # Collect statistics
        train_stats['loss_d'].append(loss_d.item())
        train_stats['loss_g'].append(loss_g.item())
        train_stats['real_score'].append(torch.mean(real_preds).item())
        train_stats['fake_score'].append(torch.mean(fake_preds).item())

    # Calculate averages for the epoch
    means = {k: np.mean(v) for k, v in train_stats.items()}
    return means['loss_g'], means['loss_d'], means['real_score'], means['fake_score']


def train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        criterion,
        device='cpu',
        save_path='/saved',
        save_images_dir='/saved/images',
        num_epochs=100
    ):
    fixed_latent = torch.randn(64, model['generator'].latent_size, 1, 1, device=device)

    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for epoch in range(num_epochs):
        # Record losses & scores
        loss_g, loss_d, real_score, fake_score = train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        fake_images = model['generator'](fixed_latent)
        fake_name = 'generated-images-{0:0=4d}.png'.format(epoch)
        
        saved_image_path = train_loader.dataset.save_images(images=fake_images, save_dir=save_images_dir, save_name=fake_name)
        print(saved_image_path)

        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, num_epochs, 
            losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1]))
        
        # torch.save(
        #     {
        #         'generator': model['generator'].state_dict(),
        #         'discriminator': model['discriminator'].state_dict(), 
        #         'optimizer': optimizer.state_dict()
        #     }, 
        #     f"{save_path}/epoch={epoch}.pth"
        # )

        ssim, fid = eval_epoch(data_loader=eval_loader, model=model, device=device)

        wandb.log({
            "loss_generator": loss_g,
            "loss_discriminator": loss_d,
            "real_score": real_score,
            "fake_score": fake_score,
            "generated_image": wandb.Image(saved_image_path),
            "SSIM": ssim,
            "FID": fid
        })
        
        print("ssim: {:.4f}  fid: {:.4f}".format(ssim, fid))