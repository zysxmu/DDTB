import json
import math
import pdb
from decimal import Decimal

import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import loss as L

import data_new
import model
import utility
from model.edsr_org import EDSR
from model.rdn_org import RDN
from model.bnsrresnet_org import SRResNet as bnSRResNet
from option import args
from utils import common as util
from utils.common import AverageMeter, load_check

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')

class Trainer():
    def __init__(self, args, loader, t_model, my_loss,ckp):

        self.epoch = 0
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = t_model
        self.loss = my_loss

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon)
        self.sheduler = StepLR(self.optimizer, step_size=int(args.decay), gamma=args.gamma)

        self.writer_train = SummaryWriter(ckp.dir + '/run/train')

        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.sheduler.load_state_dict(ckpt['scheduler'])

        self.error_last = 1e8
        self.losses = AverageMeter()


    def train(self):
        
        self.epoch = self.epoch + 1
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.writer_train.add_scalar(f'lr', lr, self.epoch)
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch, Decimal(lr))
        )

        self.model.train()

        num_iterations = len(self.loader_train)
        timer_data, timer_model = utility.timer(), utility.timer()
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):


            num_iters = num_iterations * (self.epoch - 1) + batch

            lr, hr = self.prepare(lr, hr)
            data_size = lr.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            sr = self.model(lr)
            # start log
            self.loss.start_log()
            loss = self.loss(sr, hr)
            self.loss.end_log(len(lr))
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
        
            loss.backward()
            self.optimizer.step()
            
    
            timer_model.hold()
            self.losses.update(loss.item(), data_size)
            display_loss = self.loss.display_loss(len(lr))+f'Loss: {self.losses.avg: .3f}'

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    display_loss,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()
        self.sheduler.step()


    def test(self, is_teacher=False):
        torch.set_grad_enabled(False)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        model = self.model
        model.eval()
        timer_test = utility.timer()

        if self.args.save_results: self.ckp.begin_background()
        self.savesau = {}
        self.savesal = {}
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                i = 0
                for lr, hr, filename in tqdm(d, ncols=80):
                    i += 1
                    lr, hr = self.prepare(lr, hr)
                    sr = model(lr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]
                    cur_psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += cur_psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        save_name = f'{args.k_bits}bit_{filename[0]}'
                        self.ckp.save_results(d, save_name, save_list, scale)
                # pdb.set_trace()
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}] PSNR: {:.3f}  (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1,
                    )
                )
                self.writer_train.add_scalar(f'psnr', self.ckp.log[-1, idx_data, idx_scale], self.epoch)

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            is_best = (best[1][0, 0] + 1 == self.epoch)
            state = {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.sheduler.state_dict()
            }
            util.save_checkpoint(state, is_best, checkpoint=self.ckp.dir + '/model')

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.cuda()

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs


def main():
    global checkpoint
    if checkpoint.ok:
        loader = data_new.Data(args)
        if args.model.lower() == 'edsr':
            t_model = EDSR(args, is_teacher=False).to(device)
        elif args.model.lower() == 'rdn':
            t_model = RDN(args, is_teacher=False).to(device)
        elif args.model.lower() == 'bnsrresnet':
            t_model = bnSRResNet(args, is_teacher=False).to(device)
        else:
            raise ValueError('not expected model = {}'.format(args.model))

        if args.pre_train is not None:
            tckpt = torch.load(args.pre_train)
            t_checkpoint = tckpt['state_dict'] if 'state_dict' in tckpt else tckpt
            t_model.load_state_dict(t_checkpoint)
        if args.test_only:
            if args.refine is None:
                ckpt = torch.load(f'{args.save}/model/model_best.pth.tar')
                refine_path = f'{args.save}/model/model_best.pth.tar'
            else:
                ckpt = torch.load(f'{args.refine}')
                refine_path = args.refine

            t_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

            t_model.load_state_dict(t_checkpoint)
            print(f"Load model from {refine_path}")
        _loss = L.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, t_model, _loss, checkpoint)

        print(f'{args.save} start!')
        while not t.terminate():
            # t.test(True)
            t.train()
            t.test()

        checkpoint.done()
        print(f'{args.save} done!')


if __name__ == '__main__':
    main()
