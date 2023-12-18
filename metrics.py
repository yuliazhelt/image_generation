from piq import ssim, FID
import torch
from torchvision.models import inception_v3
import torchvision.transforms as transforms

def SSIM_metric(generated_images, real_images):
    ssim_value = ssim((generated_images + 1) / 2, (real_images + 1) / 2, data_range=1.)
    return ssim_value.item()


def get_feature_vectors(inception_model, dataloader, device):
    inception_model.eval()
    features = []
    preprocess = transforms.Compose([
        transforms.Resize((299, 299))
    ])
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            images = preprocess(images)
            # Preprocess images if needed, e.g., resize, normalize
            output = inception_model(images)
            features.append(output.cpu())
    return torch.cat(features)


def FID_metric(generated_dataloader, real_dataloader, device):
    inception_model = inception_v3(pretrained=True).to(device)

    real_features = get_feature_vectors(inception_model, real_dataloader, device)
    generated_features = get_feature_vectors(inception_model, generated_dataloader, device)

    fid_metric = FID()
    fid_score = fid_metric(real_features, generated_features)
    return fid_score