from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    train_data = load_data('dense_data/train')
    valid_data = load_data('dense_data/valid')

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (img, aim_point) in enumerate(train_data):
            img, aim_point = img.to(device), aim_point.to(device)
            
            optimizer.zero_grad()
            pred_aim_point = model(img)
            loss_val = criterion(pred_aim_point, aim_point)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                log(train_logger, img, aim_point, pred_aim_point, global_step)  # logging images and aim points

            loss_val.backward()
            optimizer.step()
            
            global_step += 1
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(img)}/{len(train_data.dataset)} '
                      f'({100. * batch_idx / len(train_data):.0f}%)]\tLoss: {loss_val.item():.6f}')

        # Validation and logging for validation can go here

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--log-interval', type=int, default=10, help='Num of batches to wait before logging training status')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.3), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)


