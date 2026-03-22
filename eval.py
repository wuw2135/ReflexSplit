import os
from os.path import join

import torch
import torch.backends.cudnn as cudnn

import data.sirs_dataset as datasets
from engine import Engine
from options import SIRSOptions
from tools import mutils

opt = SIRSOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

datadir = os.path.join(opt.base_dir)




# Define evaluation/test dataset
eval_dataset_real = datasets.RealEvalDataset(join(datadir, f'real20_420'), size_rounded=opt.size_rounded)
eval_dataset_postcard = datasets.SIREvalDataset(join(datadir, 'SIR2/PostcardDataset'), size_rounded=opt.size_rounded)
eval_dataset_solid = datasets.SIREvalDataset(join(datadir, 'SIR2/SolidObjectDataset'), size_rounded=opt.size_rounded)
eval_dataset_wild = datasets.SIREvalDataset(join(datadir, 'SIR2/WildSceneDataset'), size_rounded=opt.size_rounded)
eval_dataset_nature = datasets.RealEvalDataset(join(datadir, 'Nature/test'), size_rounded=opt.size_rounded)
eval_dataset_openrr_valid = datasets.RealEvalDataset(join(datadir, 'OpenRR-1k/val_100'), size_rounded=opt.size_rounded)
eval_dataset_openrr_test = datasets.RealEvalDataset(join(datadir, 'OpenRR-1k/test_100'), size_rounded=opt.size_rounded)

eval_dataloader_real = datasets.DataLoader(eval_dataset_real, batch_size=1, shuffle=False, num_workers=opt.nThreads)
eval_dataloader_nature = datasets.DataLoader(eval_dataset_nature, batch_size=1, shuffle=False, num_workers=opt.nThreads)
eval_dataloader_solid = datasets.DataLoader(eval_dataset_solid, batch_size=1, shuffle=False, num_workers=opt.nThreads)
eval_dataloader_postcard = datasets.DataLoader(eval_dataset_postcard, batch_size=1, shuffle=False,
                                               num_workers=opt.nThreads)
eval_dataloader_wild = datasets.DataLoader(eval_dataset_wild, batch_size=1, shuffle=False, num_workers=opt.nThreads)

eval_dataloader_openrr_valid = datasets.DataLoader(eval_dataset_openrr_valid, batch_size=1, shuffle=False, prefetch_factor=32, num_workers=opt.nThreads)
engine = Engine(opt)

eval_dataloader_openrr_test = datasets.DataLoader(eval_dataset_openrr_test, batch_size=1, shuffle=False, prefetch_factor=32, num_workers=opt.nThreads)
engine = Engine(opt)




"""Main Loop"""

result_dir = os.path.join('./checkpoints', opt.name, mutils.get_formatted_time())
epoch = torch.load(opt.weight_path)['epoch']

res1 = engine.eval(eval_dataloader_real, dataset_name='eval_real',
                   savedir=join(result_dir, 'real20'))
print(res1)

res2 = engine.eval(eval_dataloader_solid, dataset_name='eval_solidobject',
                   savedir=join(result_dir, 'solidobject'))
print(res2)
res3 = engine.eval(eval_dataloader_postcard, dataset_name='eval_postcard',
                   savedir=join(result_dir, 'postcard'))
print(res3)
res4 = engine.eval(eval_dataloader_wild, dataset_name='eval_wild',
                   savedir=join(result_dir, 'wild'))
print(res4)

res5 = {}

if opt.test_nature:
    res5 = engine.eval(eval_dataloader_nature, dataset_name='eval_nature',
                       savedir=join(result_dir, 'nature'), suffix='nature')
    print(res5)

res6 = engine.eval(eval_dataloader_openrr_valid, dataset_name='eval_openrr',
                       savedir=join(result_dir, 'openrr-valid'), suffix='openrr')
print(res6)

res7 = engine.eval(eval_dataloader_openrr_test, dataset_name='eval_openrr',
                       savedir=join(result_dir, 'openrr-test'), suffix='openrr')
print(res7)

print(f"{opt.name}")

print("%.2f" % res1["PSNR"])
print("%.3f" % res1["SSIM"])

print("%.2f" % res2["PSNR"])
print("%.3f" % res2["SSIM"])

print("%.2f" % res3["PSNR"])
print("%.3f" % res3["SSIM"])

print("%.2f" % res4["PSNR"])
print("%.3f" % res4["SSIM"])

if opt.test_nature:
    print("%.2f" % res5["PSNR"])
    print("%.3f" % res5["SSIM"])


print("%.2f" % res6["PSNR"])
print("%.3f" % res6["SSIM"])


print("%.2f" % res7["PSNR"])
print("%.3f" % res7["SSIM"])