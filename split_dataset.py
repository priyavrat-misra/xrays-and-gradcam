import os
import shutil
import random

class_names = ['normal', 'pneumonia', 'covid_19', 'lung_opacity']
root_dir = 'data/COVID-19_Radiography_Dataset'
source_dirs = ['Normal', 'Viral Pneumonia', 'COVID', 'Lung_Opacity']

os.mkdir(os.path.join(root_dir, 'train'))
os.mkdir(os.path.join(root_dir, 'test'))
os.mkdir(os.path.join(root_dir, 'val'))

for i, d in enumerate(source_dirs):
    os.renames(
        os.path.join(root_dir, d),
        os.path.join(root_dir, class_names[i])
    )

for c in class_names:
    os.mkdir(os.path.join(root_dir, 'test', c))
    os.mkdir(os.path.join(root_dir, 'val', c))

    images = [i for i in os.listdir(os.path.join(
        root_dir, c)) if i.lower().endswith('png')]
    test_images = random.sample(images, 60)
    images = list(set(images) - set(test_images))
    val_images = random.sample(images, 60)

    for image in test_images:
        shutil.move(
            os.path.join(root_dir, c, image),
            os.path.join(root_dir, 'test', c, image)
        )

    for image in val_images:
        shutil.move(
            os.path.join(root_dir, c, image),
            os.path.join(root_dir, 'val', c, image)
        )

    shutil.move(
        os.path.join(root_dir, c),
        os.path.join(root_dir, 'train')
    )
