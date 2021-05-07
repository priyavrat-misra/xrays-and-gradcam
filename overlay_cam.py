import os
import argparse
import warnings
import utils
import networks
from grad_cam import GradCAM
from plot_utils import apply_mask


def get_args():
    parser = argparse.ArgumentParser(
        prog="GradCAM on Chest X-Rays",
        description="Overlays given label's CAM on a given Chest X-Ray."
    )
    parser.add_argument(
        '-i', '--image-path', type=str, default='./assets/original.jpg',
        help='Path to chest X-Ray image.'
    )
    parser.add_argument(
        '-l', '--label', type=str, default=None,
        choices=['covid_19', 'lung_opacity', 'normal', 'pneumonia'],
        help='Choose from covid_19, lung_opacity, normal & pneumonia,\n'
        'to get the corresponding CAM.\n'
        'If not mentioned, the highest scoring label is considered.'
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        choices=['vgg16', 'resnet18', 'densenet121'],
        help='Choose from vgg16, resnet18 or densenet121.'
    )
    parser.add_argument(
        '-o', '--output-path', type=str, default='./outputs/output.jpg',
        help='Format: "<path> + <file_name> + .jpg"'
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # get path of the pretrained model checkpoint
    path = {
        'vgg16': './models/lr3e-5_vgg_cuda.pth',
        'resnet18': './models/lr3e-5_resnet_cuda.pth',
        'densenet121': './models/lr3e-5_densenet_cuda.pth'
    }
    path = path[args.model]

    if not os.path.exists(path):
        raise Exception(
            f'{path} not found.\n'
            'Download the required model from the following link.\n'
            'https://drive.google.com/drive/folders/'
            '14L8wd-d2a3lvgqQtwV-y53Gsnn6Ud2-w'
        )

    # load the model using pretrained weights
    model = eval(
        f'networks.get_{args.model}(out_features=4, path="{path}")'
    ).cpu()

    # set target layer for CAM
    if args.model == 'vgg16' or args.model == 'densenet121':
        target_layer = model.features[-1]
    elif args.model == 'resnet18':
        target_layer = model.layer4[-1]

    # get given label's index
    label = {
        'covid_19': 0,
        'lung_opacity': 1,
        'normal': 2,
        'pneumonia': 3
    }
    idx_to_label = {v: k for k, v in label.items()}
    if args.label is not None:
        label = label[args.label]
    else:
        label = None

    # load and preprocess image
    image = utils.load_image(args.image_path)

    warnings.filterwarnings("ignore", category=UserWarning)
    # pass image through model and get CAM for the given label
    cam = GradCAM(model=model, target_layer=target_layer)
    label, mask = cam(image, label)
    print(f'GradCAM generated for label "{idx_to_label[label]}".')

    # deprocess image and overlay CAM
    image = utils.deprocess_image(image)
    image = apply_mask(image, mask)

    # save the image
    utils.save_image(image, args.output_path)
