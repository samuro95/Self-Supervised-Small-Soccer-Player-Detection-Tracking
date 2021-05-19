from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
import os

class TestModel():

    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        assert(not opt.isTrain)
        self.seg_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                  opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.detec_netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                  opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.load_network(self.seg_netG, 'G', opt.which_epoch, 'seg')
        self.load_network(self.detec_netG, 'G', opt.which_epoch, 'detec')

        print('---------- Networks initialized -------------')
        networks.print_network(self.seg_netG)
        networks.print_network(self.detec_netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.seg_netG(self.real_A)
            fake_B = (self.fake_B + 1.0)/2.0
            input_A = (self.real_A + 1.0)/2.0
            self.fake_C = (fake_B * input_A) * 2.0 - 1
            """
            fake_B = self.fake_B.data
            input_A = self.input_A

            # composite image for detection GAN
            fake_B = (fake_B + 1.0)/2.0  # --> [0, 1]
            input_A = (input_A + 1.0)/2.0 # --> [0, 1]
            masked_A = torch.mul(input_A, fake_B)
            masked_A = masked_A * 2.0 - 1   # normalize to [-1, 1]

            masked_A = Variable(masked_A, volatile = True) # for debug
            self.masked_A = masked_A
            """
            self.fake_D = self.detec_netG(self.fake_C)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        # same as in Pix2PixModel
        """
        self.real_A = Variable(self.input_A, volatile = True)
        self.fake_B = self.seg_netG(self.real_A)
        self.fake_C = self.detec_netG(self.real_A)
        self.real_C = Variable(self.input_B, volatile = True)
        """
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        fake_D = util.tensor2im(self.fake_D.data)
        fake_C = util.tensor2im(self.fake_C.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('fake_C', fake_C),
        ('fake_D', fake_D)])

    def load_network(self, network, network_label, epoch_label, phase_label):
        save_filename = '%s_%s_net_%s.pth' % (phase_label, epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
