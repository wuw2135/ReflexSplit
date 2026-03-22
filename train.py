import os
from os.path import join
import os.path as osp
from pathlib import Path

import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options import SIRSOptions
from tools import mutils

opt = SIRSOptions().parse()
print(opt)

opt.isTrain = True
opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1

datasets.img_size = opt.img_size

datadir = os.path.join(opt.base_dir)
datadir_syn = join(datadir, 'VOC2012/PNGImages')
datadir_real = join(datadir, 'real20/train')
datadir_nature = join(datadir, 'Nature/train')

train_dataset = datasets.DSITSynTrainDataset(
    datadir_syn, read_fns('data/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True)

train_dataset_real = datasets.DSITRealTrainDataset(datadir_real, enable_transforms=True)
train_dataset_nature = datasets.DSITRealTrainDataset(datadir_nature, enable_transforms=True)

train_dataset_fusion = datasets.FusionDataset([train_dataset,
                                               train_dataset_real,
                                               train_dataset_nature], [0.6, 0.2, 0.2],
                                              size=opt.num_train if opt.num_train > 0 else 5000)

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=True, prefetch_factor=32, num_workers=32)


eval_dataset_openrr = datasets.RealEvalDataset(join(datadir, f'OpenRR-1k/val_100'), size_rounded=True)
eval_dataloader_openrr = datasets.DataLoader(
    eval_dataset_openrr, batch_size=1, shuffle=False, prefetch_factor=32, num_workers=32)

resume_dir = None
orig_weight_path = getattr(opt, "weight_path", None)
if orig_weight_path is not None and osp.isdir(orig_weight_path):
    resume_dir = orig_weight_path
    opt.weight_path = None

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(f'./checkpoints/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

def eval_with_weights(weight_fpath, tag):
    if not osp.isfile(weight_fpath):
        print(f"[WARN] weight file not found: {weight_fpath}")
        return None

    engine.model.opt.weight_path = weight_fpath
    engine.model.load_weights()
    eval_m = engine.eval(eval_dataloader_openrr, dataset_name='testdata_openrr',
                         savedir=None, suffix='openrr', max_save_size=10)
    
    if tag == "latest":
        val_psnr = eval_m['PSNR']  
        engine.model.step(val_psnr)


if resume_dir is not None:
    best_path, latest_path = [max((p for p in Path(resume_dir).glob(f'*{s}') if p.is_file()),
                              key=lambda x: x.stat().st_mtime, default=None)
                          for s in ('best_psnr.pt', 'latest.pt')]
    eval_with_weights(best_path, tag="best")
    eval_with_weights(latest_path, tag="latest")

else:
    if opt.resume or opt.debug_eval:
        save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        engine.save_model()

        eval_m = engine.eval(eval_dataloader_openrr, dataset_name='testdata_openrr',
                    savedir=save_dir, suffix='openrr', max_save_size=10)
        val_psnr = eval_m['PSNR']  
        engine.model.scheduler_G.step(val_psnr)

print(f"[DEBUG] Current epoch: {engine.epoch}")
print(f"[DEBUG] Target epoch: {opt.nEpochs}")
print(f"[DEBUG] Will train: {engine.epoch < opt.nEpochs}")

# define training strategy
set_learning_rate(opt.lr)
while engine.epoch < opt.nEpochs:
    print('random_seed: ', opt.seed)
    engine.train(train_dataloader_fusion)

    if engine.epoch % 1 == 0:
        save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)

        eval_m = engine.eval(eval_dataloader_openrr, dataset_name='testdata_openrr',
                savedir=save_dir, suffix='openrr', max_save_size=10)

        val_psnr = eval_m['PSNR'] 
        engine.model.scheduler_G.step(val_psnr)
