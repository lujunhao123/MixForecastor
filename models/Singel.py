import torch
import torch.nn as nn

from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS,  ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer,ASHyper,ForecastGrapher,FourierGNN,Mamba


class Model(nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args
        self.model_dict = {      
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Nonstationary_Transformer': Nonstationary_Transformer,
        'DLinear': DLinear,
        'FEDformer': FEDformer,
        'Informer': Informer,
        'LightTS': LightTS,
        'ETSformer': ETSformer,
        'PatchTST': PatchTST,
        'Pyraformer': Pyraformer,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'FiLM': FiLM,
        'iTransformer': iTransformer,
        'Koopa': Koopa,
        'TiDE': TiDE,
        'FreTS': FreTS,
        'MambaSimple': MambaSimple,
        'TimeMixer': TimeMixer,
        'TSMixer': TSMixer,
        'SegRNN': SegRNN,
        'TemporalFusionTransformer': TemporalFusionTransformer,
        "SCINet": SCINet,
        'PAttn': PAttn,
        'TimeXer': TimeMixer,
        'WPMixer': WPMixer,
        'MultiPatchFormer': MultiPatchFormer,
        'ASHyper' : ASHyper,
        'ForecastGrapher' : ForecastGrapher,
        'FourierGNN' : FourierGNN,
        'Mamba':Mamba,
        }

        self.backbone = self.model_dict[args.model].Model(args)
        self.con_task_predictor = nn.Sequential(
            nn.Linear(args.pred_len,args.pred_len)
        )
        self.dis_task_predictor = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )



    def forward(self, batch_x_dis, batch_x_con, batch_x_mark, batch_y_dis, batch_y_con, batch_y_mark):
        B,N,L = batch_x_con.shape
        x, x_mark_enc, x_dec, x_mark_dec = batch_x_con, batch_x_mark, batch_y_con, batch_y_mark
        x, x_mark_enc, x_dec, x_mark_dec = x.transpose(-1, -2), x_mark_enc.transpose(-1, -2), x_dec.transpose(-1, -2), x_mark_dec.transpose(-1, -2)
        feature = self.backbone(x, x_mark_enc, x_dec, x_mark_dec)

        output = feature.transpose(-1, -2)
        if self.args.Singel_task == "Con":
            output = self.con_task_predictor(output)
        elif self.args.Singel_task == 'Dis':
            output = self.dis_task_predictor(output[:,:,:,None])
        return output

