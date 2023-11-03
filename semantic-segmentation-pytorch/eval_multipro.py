# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import openpyxl
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color101.mat')['colors']
#
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def visualize_result(data, pred, dir_result, gt_dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    # im_vis = np.concatenate((img, seg_color, pred_color),
    #                         axis=1).astype(np.uint8)

    im_vis = pred_color.astype(np.uint8)
    gt_im_vis = seg_color.astype(np.uint8)

    img_name = info.split('/')[-1]
    print(img_name)
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))
    Image.fromarray(gt_im_vis).save(os.path.join(gt_dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue):
    segmentation_module.eval()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue.put_nowait((acc, pix, intersection, union))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result'),
                os.path.join(cfg.DIR, 'gt_result')
            )


def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)

def compare_result(input_dir_original,input_dir_result,input_dir_gt,output_dir):
    # get the filename in the directory
    original_images = os.listdir(input_dir_original)
    result_images = os.listdir(input_dir_result)
    semantic_images = os.listdir(input_dir_gt)

    # ensure the pattern is the same
    original_images.sort()
    result_images.sort()
    semantic_images.sort()

    # create the output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # combine the image
    for i in range(len(original_images)):
        original_image = Image.open(os.path.join(input_dir_original, original_images[i]))
        print("Dealing with:"+ os.path.join(input_dir_original, original_images[i]))
        result_image = Image.open(os.path.join(input_dir_result, result_images[i]))
        semantic_image = Image.open(os.path.join(input_dir_gt, semantic_images[i]))

        # check the size of the image is match or not
        if original_image.size == result_image.size == semantic_image.size:
            # create a new image
            combined_image = Image.new('RGB', (original_image.width * 3, original_image.height))
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(result_image, (original_image.width, 0))
            combined_image.paste(semantic_image, (original_image.width * 2, 0))

            # save the conbine image
            combined_image.save(os.path.join(output_dir, original_images[i].replace('.jpg', '_combined.jpg')))

    # Finish
    print('Processed Finish!')


def main(cfg, gpus):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    
    # specify
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
    
    # mean iou
    new_iou = []
    wb = openpyxl.load_workbook('apartment0_classes.xlsx', data_only=True)
    s1 = wb['Sheet1']
    for i in range(2,51):
        label = s1.cell(i,1).value
        print("add  " + str(int(label)) + "  " +  str(iou[int(label)]))
        new_iou.append(iou[int(label)])
    wb.save('apartment0_classes.xlsx')

    iou_count = 0
    iou_sum = 0
    for iou in new_iou:
        if iou != 0:
            iou_count += 1
            iou_sum += iou

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
            .format(iou_sum/iou_count, acc_meter.average()*100))

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
