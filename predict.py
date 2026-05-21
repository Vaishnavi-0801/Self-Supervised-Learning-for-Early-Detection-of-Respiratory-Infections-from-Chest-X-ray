import torch
import torchvision.models as models
import numpy as np
import cv2
import h5py

model = models.resnet18(weights=None)
model.fc = torch.nn.Identity()

with h5py.File("model/ssl_model.h5", "r") as f:
    state_dict = model.state_dict()

    for k in state_dict.keys():
        if k in f:
            state_dict[k] = torch.tensor(f[k][()])

    model.load_state_dict(state_dict)

model.eval()


def extract_features(img_path):

    # Read image
    img = cv2.imread(img_path)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Change HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Convert to tensor
    img = torch.tensor(img).float().unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(img)

    features = features.numpy().flatten()

    feature_dim = len(features)

    # Values for graph
    chart_values = features[:10].tolist()

    return feature_dim, chart_values, features