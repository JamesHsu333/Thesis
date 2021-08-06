import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns; 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import dataloaders.dataloader as dataloader
from model.deeplab.deeplab import *
from model.attention.anet import *
from model.FCN.FCN import *
from model.GCN.GCN import *
from model.sync_batchnorm.replicate import patch_replication_callback
import utils.utils as utils

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
        help="Directory containing params.json")
parser.add_argument('--model_type', default='deeplab',
        help="Type of deeplab")
parser.add_argument('--num_classes', default=21,
        help="Numbers of classes")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
        containing weights to load")
sns.set_theme()

def evaluate_save_images(model, model_type, dataloader, params):
    model.eval()
    i=1

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for sample in dataloader:
            data_batch, labels_batch = sample['image'], sample['label']
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

            with torch.no_grad():
                output_batch, fm4, fm4_out = model(data_batch)

            output_batch = output_batch.data.cpu().numpy()
            data_batch = data_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            fm4 = fm4.data.cpu().numpy()
            fm4_out = fm4_out.data.cpu().numpy()

            output_batch_argmax = np.argmax(output_batch, axis=1)

            save_mask(labels_batch, model_type, i)
            save_prediction(output_batch_argmax, model_type, i)
            """
            if i==81:
                save_heat_maps(fm4, fm4_out, output_batch, i)
                break
            """
            i+=1
            t.update()
    return

# Decode Segmentation Mask
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Save mask
def save_mask(labels_images, model_type, index):
    for _, label_image in enumerate(labels_images):
        im = Image.fromarray(decode_segmap(label_image))
        im.save("images\{}\mask\{}.png".format(model_type, index))

# Save images
def save_prediction(output_images, model_type, index):
    for _, output_image in enumerate(output_images):
        im = Image.fromarray(decode_segmap(output_image))
        im.save("images\{}\prediction\{}.png".format(model_type, index))

# Save heat maps
def save_heat_maps(fms, fm_outs, outs, index):
    for _, (fm, fm_out, out) in enumerate(zip(fms, fm_outs, outs)):
        for i, map in enumerate(fm):
            if i%100==1:
                sns.heatmap(map, cbar=None)
                plt.show()
        for fm_map, out_map in zip(fm_out, out):
            sns.heatmap(fm_map)
            plt.show()
            sns.heatmap(out_map)
            plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
            json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    params.batch_size = 1

    torch.manual_seed(1)
    if params.cuda:
        torch.cuda.manual_seed(1)

    utils.set_logger(os.path.join(args.model_dir, 'save_img.log'))

    logging.info("Creating the dataset...")

    dl = dataloader.fetch_dataloader(['val'], params)

    test_dl = dl['val']

    logging.info("-done")

    if args.model_type=='ANet':
        model = ANet(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    elif args.model_type=='ANet_without_filter':
        model = ANet_without_filter(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    elif args.model_type=='FCN':
        model = FCN(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    elif args.model_type=='GCN':
        model = GCN(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    elif args.model_type=='GCN_C':
        model = GCN_C(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)
    else:
        model = DeepLab(num_classes=args.num_classes,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

    if params.cuda:
        model = nn.DataParallel(model, device_ids=[0])
        patch_replication_callback(model)
        model = model.cuda()

    logging.info("-Model Type: {}".format(args.model_type))

    logging.info("Saving image")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.model_type + '_' + args.restore_file + '.pth.tar'), model)

    # Save images
    evaluate_save_images(model, args.model_type, test_dl, params)

    logging.info("- done.")