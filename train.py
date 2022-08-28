# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='./coco2014/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='./wikiart/train',
                    help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', 
                    help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./exp',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--stage1_iter', type=int, default=80000)
parser.add_argument('--stage2_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=5.0) 
parser.add_argument('--identity_weight', type=float, default=50.0) 
parser.add_argument('--AR1_weight', type=float, default=0.5) 
parser.add_argument('--AR2_weight', type=float, default=500.0) 
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--resume', action='store_true', help='enable it to train the model from checkpoints')
parser.add_argument('--checkpoints', default='./checkpoints',
                    help='Directory to save the training checkpoints')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
checkpoints_dir = Path(args.checkpoints)
checkpoints_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

discriminator = net.AesDiscriminator()

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, discriminator)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()}], lr=args.lr)
optimizer_D = torch.optim.Adam(network.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

start_iter = -1

# Enable it to train the model from checkpoints
if args.resume:
    checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
    network.load_state_dict(checkpoints['net'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    start_iter = checkpoints['epoch']

# Training
for i in range(start_iter+1, args.stage1_iter+args.stage2_iter):
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_D, iteration_count=i) 
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    
    if i < args.stage1_iter:
        stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_id, _ = network(content_images, style_images)
    else:
        stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_AR1, loss_AR2 = network(content_images, style_images, aesthetic=True)

    # train discriminator
    optimizer_D.zero_grad()
    loss_gan_d.backward(retain_graph=True)


    # train generator
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s

    loss_gan_g = args.gan_weight * loss_gan_g

    if i < args.stage1_iter:
        loss_id = args.identity_weight * loss_id
        loss = loss_c + loss_s + loss_gan_g + loss_id
    else:
        loss_AR1 = args.AR1_weight * loss_AR1
        loss_AR2 = args.AR2_weight * loss_AR2
        loss = loss_c + loss_s + loss_gan_g + loss_AR1 + loss_AR2

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer_D.step()
    
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)
    writer.add_scalar('loss_gan_g', loss_gan_g.item(), i + 1)
    writer.add_scalar('loss_gan_d', loss_gan_d.item(), i + 1)

    # Save intermediate results
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    if (i + 1) % 500 == 0:
        visualized_imgs = torch.cat([content_images, style_images, stylized_results])
      
        output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
        save_image(visualized_imgs, str(output_name), nrow=args.batch_size)
        print('[%d/%d] loss_content:%.4f, loss_style:%.4f, loss_gan_g:%.4f, loss_gan_d:%.4f' % (i+1, args.stage1_iter+args.stage2_iter, loss_c.item(), loss_s.item(), loss_gan_g.item(), loss_gan_d.item()))

    # Save models
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.stage1_iter+args.stage2_iter:
        checkpoints = {
            "net": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": i
        }
        torch.save(checkpoints, checkpoints_dir / 'checkpoints.pth.tar')
        
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth'.format(i + 1))

        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'transformer_iter_{:d}.pth'.format(i + 1))

        state_dict = network.discriminator.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'discriminator_iter_{:d}.pth'.format(i + 1))

writer.close()
