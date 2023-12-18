import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import os
import wandb
from copy import copy
import time

from dataset import CustomDataset
from dcgan import Generator, Discriminator
from trainer import train


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)

    config_path = '/home/ubuntu/image_generation/configs/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Generate a unique identifier using the current timestamp
    unique_id = time.strftime("%m%d-%H%M%S")

    wandb.init(
        project="image generation",
        name=f"DCGAN_{unique_id}",
        config=config
    )

    # data
    dataset = CustomDataset(**config['dataset']['train'])

    # dataset.show_batch()
    dataloader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    
    print("len train set", len(dataset))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = {
        "discriminator": Discriminator(**config['model']['discriminator']).to(device),
        "generator": Generator(**config['model']['generator']).to(device)
    }
    print(device)
    print(model)

    optimizer = {
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(), 
                                          lr=config['train']['lr'], betas=config['train']['betas']),
        "generator": torch.optim.Adam(model["generator"].parameters(),
                                      lr=config['train']['lr'], betas=config['train']['betas'])
    }

    criterion = {
        "discriminator": nn.BCELoss(),
        "generator": nn.BCELoss()
    }

    # load_path = config['train']['load_path']
    # if load_path != 'none':
    #     model_dict = torch.load(load_path)
    #     model.load_state_dict(model_dict['model'])
    #     optimizer.load_state_dict(model_dict['optimizer'])
    #     print("model loaded")

    os.mkdir(f"{config['train']['save_images_dir']}/run_{unique_id}")

    train(
        model=model,
        train_loader=dataloader, 
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config['train']['num_epochs'],
        device=device,
        # save_path=f"{config['train']['save_path']}/run_{unique_id}",
        save_images_dir=f"{config['train']['save_images_dir']}/run_{unique_id}"
    )
    wandb.finish()