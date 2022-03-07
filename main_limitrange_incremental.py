import json
import math
import pdb
from decimal import Decimal

import cv2
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# import data
import data_new
import model
import utility
from model.edsr import DDTB_EDSR
from model.edsr_org import EDSR
from model.rdn import DDTB_RDN
from model.rdn_org import RDN
from model.bnsrresnet import DDTB_SRResNet
from model.bnsrresnet_org import SRResNet as bnSRResNet

from option import args
from utils import common as util
from utils.common import AverageMeter, load_check
from model.quant_ops import DDTB_quant_act_asym_dynamic_quantized
from model.quant_ops import conv3x3, quant_weight, quant_weight_asym99
import numpy as np
from torch.backends import cudnn
import random

torch.manual_seed(args.seed)
cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(args.seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)

checkpoint = utility.checkpoint(args)
device = torch.device('cpu' if args.cpu else f'cuda:{args.gpu_id}')

class Trainer():
    def __init__(self, args, loader, t_model, s_model, ckp):
        self.args = args
        self.scale = args.scale

        self.epoch = 0
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.t_model = t_model
        self.s_model = s_model
        arch_param = [v for k, v in self.s_model.named_parameters() if 'alpha' not in k]
        alpha_param = [v for k, v in self.s_model.named_parameters() if 'alpha' in k]
        a = [k for k, v in self.s_model.named_parameters() if 'alpha' in k]

        params = [{'params': arch_param}, {'params': alpha_param, 'lr': 1e-2}]
        self.optimizer = torch.optim.Adam(params, lr=args.lr, betas=args.betas, eps=args.epsilon)
        self.sheduler = StepLR(self.optimizer, step_size=int(args.decay), gamma=args.gamma)
        self.writer_train = SummaryWriter(ckp.dir + '/run/train')

        if args.resume is not None:
            ckpt = torch.load(args.resume)
            self.epoch = ckpt['epoch']
            print(f"Continue from {self.epoch}")
            self.s_model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.sheduler.load_state_dict(ckpt['scheduler'])

        self.losses = AverageMeter()
        self.att_losses = AverageMeter()
        self.nor_losses = AverageMeter()

        # if not test
        forcheck = 0
        if not args.test_only:
            # set not quantized for now
            self.s_model.eval()
            for n, m in self.s_model.named_modules():
                if isinstance(m, quant_weight) or isinstance(m, quant_weight_asym99) \
                        or isinstance(m, DDTB_quant_act_asym_dynamic_quantized):
                    if getattr(m, 'k_bits') == args.k_bits:
                        setattr(m, 'k_bits', 32)
                    forcheck += 1
        print('forcheck', forcheck)

        self.save_output = {}
        for n, m in self.s_model.named_modules():
            if 'atten_c' in n and '.convs.0.act.' not in n and n[-1] == 'c' and n[-2] == '_':
                m.register_forward_hook(self.hook_fn_forward(n))

        self.first_stage_epoch = 6
        self.s_model.apply(lambda m: setattr(m, 'first_stage_epoch', self.first_stage_epoch))
        self.dynamic_ratio = args.dynamic_ratio
        print('dynamic_ratio', self.dynamic_ratio)

    def hook_fn_forward(self, name):
        def hook(module, input, output):
            self.save_output[name] = output
        return hook

    def train(self):

        self.epoch = self.epoch + 1
        # calibration is over
        if self.epoch == 2:
            # reset
            self.ckp = utility.checkpoint(args)
            forcheck = 0
            self.s_model.train()
            for n, m in self.s_model.named_modules():
                if isinstance(m, quant_weight) or isinstance(m, quant_weight_asym99) \
                        or isinstance(m, DDTB_quant_act_asym_dynamic_quantized):
                    # set quantized for now
                    if getattr(m, 'k_bits') == 32:
                        setattr(m, 'k_bits', args.k_bits)
                    forcheck += 1
            print(forcheck)

            var_list = []
            for n, m in self.s_model.named_modules():
                if isinstance(m, DDTB_quant_act_asym_dynamic_quantized):
                    # set open the dynamic
                    if hasattr(m, 'fp_max_list') and len(m.fp_max_list) > 0:
                        var_list.append((n, np.var(m.fp_max_list)+np.var(m.fp_min_list)))
                    m.fp_max_list.clear()
                    m.fp_min_list.clear()
            var_list.sort(key=lambda x:x[1], reverse=True)
            print(var_list)
            # random.shuffle(var_list)
            dynamice_names = set()
            for i in range(int(len(var_list)*self.dynamic_ratio)):
                dynamice_names.add(var_list[i][0])
            print('dynamice_names', dynamice_names)
            print('len dynamice_names,', len(dynamice_names), 'len var_list,', len(var_list))
            for n, m in self.s_model.named_modules():
                if n in dynamice_names:
                    setattr(m, 'is_dynamic', torch.ones(1).cuda())

        # incremental epoch is over
        if self.epoch == self.first_stage_epoch:
            # reset
            self.ckp = utility.checkpoint(args)

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        self.writer_train.add_scalar(f'lr', lr, self.epoch)
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(self.epoch, Decimal(lr))
        )

        self.t_model.eval()
        if self.epoch > 1:
            self.s_model.train()

        self.s_model.apply(lambda m: setattr(m, 'epoch', self.epoch))

        num_iterations = len(self.loader_train)
        timer_data, timer_model = utility.timer(), utility.timer()


        for batch, (lr, hr, _,) in enumerate(self.loader_train):

            num_iters = num_iterations * (self.epoch - 1) + batch

            lr, hr = self.prepare(lr, hr)
            data_size = lr.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            if hasattr(self.t_model, 'set_scale'):
                self.t_model.set_scale(idx_scale)
            if hasattr(self.s_model, 'set_scale'):
                self.s_model.set_scale(idx_scale)

            if self.epoch == 1:
                with torch.no_grad():
                    s_sr, s_res = self.s_model(lr)
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('Calibration! Accumulate the max_v and min_v for computing Variance')
            else:
                self.save_output.clear()
                with torch.no_grad():
                    t_sr, t_res = self.t_model(lr)
                s_sr, s_res = self.s_model(lr)

                nor_loss = args.w_l1 * F.l1_loss(s_sr, hr)
                att_loss = args.w_at * util.at_loss(s_res, t_res)

                if self.epoch < self.first_stage_epoch:
                    reg_loss = torch.zeros(1).cuda()
                    if len(self.save_output) > 0:
                        for n in self.save_output:
                            reg_loss += F.mse_loss(self.save_output[n], torch.zeros(self.save_output[n].shape).cuda())
                        reg_loss = reg_loss/len(self.save_output)
                    loss = nor_loss + att_loss + reg_loss
                else:
                    loss = nor_loss + att_loss

                if torch.any(torch.isnan(loss)):
                    print('None loss!!')
                    import IPython
                    IPython.embed()

                loss.backward()
                self.optimizer.step()

                timer_model.hold()

                self.losses.update(loss.item(), data_size)
                display_loss = f'Loss: {self.losses.avg: .3f}'

                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        display_loss,
                        timer_model.release(),
                        timer_data.release()))
                    if self.epoch < self.first_stage_epoch:
                        self.ckp.write_log('reg_loss:'+str(round(reg_loss.cpu().item(), 4)))
            timer_data.tic()

            for name, value in self.s_model.named_parameters():
                if 'alpha' in name:
                    # if value.grad is not None:
                    if value.grad is not None and value.grad.squeeze().ndim == 0:
                        self.writer_train.add_scalar(f'{name}_grad', value.grad.cpu().data.numpy(), num_iters)
                        self.writer_train.add_scalar(f'{name}_data', value.cpu().data.numpy(), num_iters)

        self.sheduler.step()


    def test(self, is_teacher=False):
        torch.set_grad_enabled(False)

        self.s_model.apply(lambda m: setattr(m, 'test_only', args.test_only))
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        if is_teacher:
            model = self.t_model
        else:
            model = self.s_model
        model.eval()
        timer_test = utility.timer()

        forcheck = 0
        all = 0
        for n, m in model.named_modules():
            if isinstance(m, DDTB_quant_act_asym_dynamic_quantized) and hasattr(m, 'is_dynamic'):
                if m.is_dynamic:
                    forcheck += 1
                all+=1
        print(forcheck, all)

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
                    sr, s_res = model(lr)

                    for n, m in model.named_modules():
                        if isinstance(m, DDTB_quant_act_asym_dynamic_quantized):
                            if n not in self.savesau:
                                self.savesau[n] = [0]
                                self.savesal[n] = [0]

                            if hasattr(m, 'sau'):
                                self.savesau[n][0] = round(m.alpha_upper.cpu().item(), 3)
                                self.savesal[n][0] = round(m.alpha_lower.cpu().item(), 3)

                                self.savesau[n].append(round(m.sau.cpu().item(), 3))
                                self.savesal[n].append(round(m.sal.cpu().item(), 3))

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

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)

                self.ckp.write_log(
                    '[{} x{}] PSNR: {:.3f}  (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                self.writer_train.add_scalar(f'psnr', self.ckp.log[-1, idx_data, idx_scale], self.epoch)

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            # because the after the first epoch, the ckp re-initialize
            is_best = (best[1][0, 0] + self.first_stage_epoch == epoch)
            state = {
                'epoch': epoch,
                'state_dict': self.s_model.state_dict(),
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


    if checkpoint.ok:
        loader = data_new.Data(args)
        if args.model.lower() == 'edsr':
            t_model = EDSR(args, is_teacher=True).to(device)
            s_model = DDTB_EDSR(args, bias=True).to(device)
        elif args.model.lower() == 'rdn':
            t_model = RDN(args, is_teacher=True).to(device)
            s_model = DDTB_RDN(args).to(device)
        elif args.model.lower() == 'bnsrresnet':
            t_model = bnSRResNet(args,is_teacher=True).to(device)
            s_model = DDTB_SRResNet(args).to(device)
        else:
            raise ValueError('not expected model = {}'.format(args.model))

        if args.pre_train is not None:
            t_checkpoint = torch.load(args.pre_train)
            t_checkpoint = t_checkpoint['state_dict'] if 'state_dict' in t_checkpoint else t_checkpoint
            t_model.load_state_dict(t_checkpoint)

            # quantized model load pre-train weighs

            s_model_dict = s_model.state_dict()
            pre_trained_dict = {}
            for k, v in t_checkpoint.items():
                if args.model.lower() == 'edsr':
                    if k in s_model_dict:
                        pre_trained_dict[k] = v
                    elif k.replace('.body.2', '.body.3') in s_model_dict:
                        pre_trained_dict[k.replace('.body.2', '.body.3')] = v
                    else:
                        print(k)
                else:
                    if k in s_model_dict:
                        pre_trained_dict[k] = v
                    else:
                        print(k)
            # check all pre-train parameter are loaded
            for k in pre_trained_dict:
                if args.model.lower() == 'edsr':
                    if k not in s_model_dict and k.replace('.body.2', '.body.3') not in s_model_dict:
                        print(k)
                else:
                    if k not in s_model_dict:
                        print(k)

            assert len(pre_trained_dict) == len(t_model.state_dict())
            print(len(pre_trained_dict), len(s_model_dict))
            s_model_dict.update(pre_trained_dict)
            s_model.load_state_dict(s_model_dict)

        if args.test_only:
            if args.refine is None:
                ckpt = torch.load(f'{args.save}/model/model_best.pth.tar')
                refine_path = f'{args.save}/model/model_best.pth.tar'
            else:
                ckpt = torch.load(f'{args.refine}')
                refine_path = args.refine

            s_checkpoint = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            for k, v in s_checkpoint.items():
                if 'max_val' in k or 'min_val' in k:
                    s_checkpoint[k] = torch.reshape(v, torch.ones(1).shape)
            s_model.load_state_dict(s_checkpoint)
            print(f"Load model from {refine_path}")

        t = Trainer(args, loader, t_model, s_model, checkpoint)
        # t.test()
        print(f'{args.save} start!')
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()
        print(f'{args.save} done!')


if __name__ == '__main__':
    main()
