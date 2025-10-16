import os
import torch
from models import Multi_task as M
from models import Singel as S
from models import Multi_taskm as CM

class Exp_Basic(object):
    def __init__(self, args):

        if args.Ramp_name == 'MixRamp':
            Multi_task = M
#        elif args.Ramp_name == 'AuxMitsRampX':
#            Multi_task = XM
        elif args.Ramp_name == 'SNoramlRamp':
            Multi_task = S
        elif args.Ramp_name == 'MNormalRamp':
            Multi_task = CM
        else:
            Multi_task = None

        self.args = args
        self.model_dict = {
            'TimesNet': Multi_task,
            'Autoformer': Multi_task,
            'Transformer': Multi_task,
            'Nonstationary_Transformer': Multi_task,
            'DLinear': Multi_task,
            'FEDformer': Multi_task,
            'Informer': Multi_task,
            'LightTS': Multi_task,
            'ETSformer': Multi_task,
            'PatchTST': Multi_task,
            'Pyraformer': Multi_task,
            'MICN': Multi_task,
            'Crossformer': Multi_task,
            'FiLM': Multi_task,
            'iTransformer': Multi_task,
            'Koopa': Multi_task,
            'TiDE': Multi_task,
            'FreTS': Multi_task,
            'MambaSimple': Multi_task,
            'TimeMixer': Multi_task,
            'TSMixer': Multi_task,
            'SegRNN': Multi_task,
            'TemporalFusionTransformer': Multi_task,
            "SCINet": Multi_task,
            'PAttn': Multi_task,
            'TimeXer': Multi_task,
            'WPMixer': Multi_task,
            'MultiPatchFormer': Multi_task,
            'ForecastGrapher' : Multi_task,
            'FourierGNN' : Multi_task,
            'Mamba':Multi_task,
            'TVNet':Multi_task
                    }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Multi_task

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
