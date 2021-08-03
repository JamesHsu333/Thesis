import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import dataloaders.dataloader as dataloader
from evaluate import evaluate
from model.attention.anet import *
from model.deeplab.deeplab import *
from model.FCN.FCN import *
from model.GCN.GCN import *
from model.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import loss_fns
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.summary import TensorboardSummary
import utils.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Directory containing params.json")
parser.add_argument('--num_classes', default=21,
                    help="Numbers of classes")
parser.add_argument('--model_type', default='deeplab',
                    help="Type of deeplab")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")


def train(model, dataloader, optimizer, loss_fns, scheduler, evaluator, writer, epoch, params):
    # Set model to training mode
    model.train()

    evaluator.reset()
    # Summary for current training loop and a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, sample in enumerate(dataloader):
            train_batch, labels_batch = sample['image'], sample['label']
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()

            current_lr = scheduler(optimizer, i, epoch)
            optimizer.zero_grad()

            # Forward
            output_batch, _, _ = model(train_batch)

            # Backward
            loss = loss_fns['CrossEntropy'](params, output_batch, labels_batch)

            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                output_batch = np.argmax(output_batch, axis=1)

                evaluator.add_batch(labels_batch, output_batch)

            # update the average loss
            loss_avg.update(loss.item())

            # tensorboard summary
            writer.add_scalar('train/total_loss_iter',
                              loss.item(), i + len(dataloader) * epoch)

            t.set_postfix(loss='{:05.3f}'.format(
                loss_avg()), lr='{:05.5f}'.format(current_lr))
            t.update()

    # compute mean of all metrics in summary
    writer.add_scalar('train/mean_loss_epoch', loss_avg(), epoch)
    metrics_mean = {
        'mIOU': evaluator.Mean_Intersection_over_Union(), 'loss': loss_avg()}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fns, scheduler, evaluator, writer, params, model_dir, name,
                       restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.model_type + '_' + args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer=None)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, train_dataloader, optimizer, loss_fns,
              scheduler, evaluator, writer, epoch, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dataloader,
                               loss_fns, evaluator, writer, epoch, params)

        val_acc = val_metrics['mIOU']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir,
                              name=name)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    torch.manual_seed(1)
    if params.cuda:
        torch.cuda.manual_seed(1)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    dl = dataloader.fetch_dataloader(['train', 'val'], params)

    train_dl = dl['train']
    val_dl = dl['val']

    logging.info("-done")

    if args.model_type=='ANet':
        model = ANet(num_classes=args.num_classes,
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


    logging.info("-Model Type: {}".format(args.model_type))

    train_params = [{'params': model.get_1x_lr_params(), 'lr': params.learning_rate},
                    {'params': model.get_10x_lr_params(), 'lr': params.learning_rate * 10}]

    optimizer = optim.SGD(train_params, momentum=params.momentum,
                          weight_decay=params.weight_decay)

    if params.cuda:
        model = nn.DataParallel(model, device_ids=[0])
        patch_replication_callback(model)
        model = model.cuda()

    scheduler = LR_Scheduler("poly", params.learning_rate,
                             params.num_epochs, len(train_dl))

    loss_fns = loss_fns

    # Define Tensorboard Summary
    summary = TensorboardSummary(args.model_dir)
    writer = summary.create_summary()

    evaluator = Evaluator(20+1)

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fns, scheduler, evaluator,
                       writer, params, args.model_dir, args.model_type, args.restore_file)
