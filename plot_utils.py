import pandas as pd
import matplotlib.pyplot as plt


def plot_summary(path):
    data = pd.read_csv(path)
    filename = path.split('/')[-1][:-4]

    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

    ax1.plot(data['epoch'], data['train_loss'], marker='.', label='Train Loss')
    ax1.plot(data['epoch'], data['valid_loss'], marker='.', label='Valid Loss')

    ax2.plot(data['epoch'], data['train_acc'],
             marker='.', label='Train Accuracy')
    ax2.plot(data['epoch'], data['valid_acc'],
             marker='.', label='Valid Accuracy')

    ax1.legend()
    ax1.set_title('Running Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.legend()
    ax2.set_title('Running Accuracy', fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(f'outputs/finetune_plots/{filename}.png')
    plt.show()
    plt.close()
