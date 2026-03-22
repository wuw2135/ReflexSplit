import os
import time
from os.path import join

import torch
import torch.nn as nn

import util.util as util
import util.index as index
from models import make_model
from util.visualizer import Visualizer
import numpy as np
from util.schedulers import DifferentialLambdaScheduler

from torch.utils.tensorboard import SummaryWriter
class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        os.makedirs(self.basedir, exist_ok=True)

        opt = self.opt

        """Model"""
        self.model = make_model(self.opt.model)()
        self.model.initialize(opt)
        self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs')) if not opt.no_log else None
        self.visualizer = Visualizer(opt)
        
        # ===== 新增：初始化 Lambda Scheduler =====
        self.lambda_scheduler = DifferentialLambdaScheduler(
            warmup_epochs=opt.lambda_warmup_epochs,
            max_epochs=opt.max_epochs
        )
        print(f"\n[Lambda Scheduler] Initialized with warmup_epochs={opt.lambda_warmup_epochs}, max_epochs={opt.max_epochs}")
        
    def _log_learning_rate(self):
        if self.writer and hasattr(self.model, 'optimizers'):
            for i, optimizer in enumerate(self.model.optimizers):
                for j, param_group in enumerate(optimizer.param_groups):
                    lr = param_group['lr']
                    self.writer.add_scalar(f"Learning_Rate/optimizer_{i}_group_{j}", lr, self.iterations)

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)

        lambda_info = self.lambda_scheduler.apply_to_model(self.model.network, self.epoch)
        print(f"[Lambda Scheduler] Epoch {self.epoch}: scale={lambda_info['scale']:.3f}, "
            f"updated {lambda_info['updated_modules']} modules")
    
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)

            errors = model.get_current_errors()
            avg_meters.update(errors)
            
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)
                
                self._log_learning_rate()

                if i == 0:
                    self.writer.add_scalar('Lambda/scale', lambda_info['scale'], epoch)

                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)

            self.iterations += 1

        self.epoch += 1

        if not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                      (self.epoch, self.iterations))
                model.save()

            print('saving the latest model at the end of epoch %d, iters %d' %
                  (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                  (time.time() - epoch_start_time))

        try:
            train_loader.reset()
        except:
            pass

    def eval(self, val_loader, dataset_name, savedir='./tmp', loss_key=None, max_save_size=None, **kwargs):
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            self.f = open(os.path.join(savedir, 'metrics.txt'), 'w+')
            self.f.write("name,PSNR,SSIM" + '\n')
            
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if opt.selected and data['fn'][0].split('.')[0] not in opt.selected:
                    continue
                if max_save_size is not None and i > max_save_size:
                    index = model.eval(data, savedir=None, **kwargs)
                else:
                    index = model.eval(data, savedir=savedir, **kwargs)

                if savedir is not None:
                    self.f.write(f"{data['fn'][0]},{index['PSNR']},{index['SSIM']}\n")
                    
                avg_meters.update(index)
                util.progress_bar(i, len(val_loader), str(avg_meters))

            if not opt.no_log and self.writer:
                util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)

            if 'PSNR' in avg_meters.keys():
                current_psnr = avg_meters['PSNR']
                save_dir = os.path.join('checkpoints', self.opt.name)
                os.makedirs(save_dir, exist_ok=True)

                best_path = os.path.join(save_dir, 'best_psnr.pth')
                latest_path = os.path.join(save_dir, 'latest.pth')

                self.model.save(label='latest')

                if not hasattr(self, 'best_psnr') or current_psnr > self.best_psnr:
                    self.best_psnr = current_psnr
                    print(f"[Eval] 🎯 New best PSNR: {current_psnr:.3f} at epoch {self.epoch}")
                    self.model.save(label='best_psnr')

                for f in os.listdir(save_dir):
                    if f.endswith('.pth') and f not in ('best_psnr.pth', 'latest.pth'):
                        try:
                            os.remove(os.path.join(save_dir, f))
                        except Exception as e:
                            print(f"⚠️ Failed to remove {f}: {e}")


        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    def save_model(self):
        self.model.save()

    def save_eval(self, label):
        self.model.save_eval(label)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e