import torch
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_vgg16(pretrained=False, out_features=None, path=None):
    model = torchvision.models.vgg16(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Sequential(
            *list(model.classifier.children())[:-1],
            torch.nn.Linear(in_features=4096, out_features=out_features)
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def get_resnet18(pretrained=False, out_features=None, path=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    if out_features is not None:
        model.fc = torch.nn.Linear(in_features=512, out_features=out_features)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def get_densenet121(pretrained=False, out_features=None, path=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    if out_features is not None:
        model.classifier = torch.nn.Linear(
            in_features=1024, out_features=out_features
        )
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)
