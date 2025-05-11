import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models
from .backbone import Encoder, Decoder, Bottleneck


CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}


def Soft_aggregation(ps, max_obj):
    
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj+1, H, W).to(ps.device)
    em[0, 0, :, :] =  torch.prod(1-ps, dim=0) # bg prob
    em[0,1:num_objects+1, :, :] = ps # obj prob
    em = torch.clamp(em, 1e-7, 1-1e-7)
    logit = torch.log((em /(1-em)))

    return logit


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 


class Encoder_M(nn.Module):
    def __init__(self, arch):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_bg):
        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024

        return r4, r3, r2, c1


class Encoder_Q(nn.Module):
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = in_f

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, inplane, mdim, expand):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128 * expand, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(64 * expand, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, f):
        x = self.convFM(r4)
        m4 = self.ResMM(x)
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False)
        return p


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape

        qi = q_in.reshape(-1, C, H*W)
        p = torch.bmm(m_in, qi) # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1) # no x centers x hw

        mo = m_out.permute(0, 2, 1) # no x c x centers 
        mem = torch.bmm(mo, p) # no x c x hw
        mem = mem.reshape(no, vd, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)


class MetaClassifier(nn.Module):

    def __init__(self, channels_in, channels_mem):
        super(MetaClassifier, self).__init__()
        self.cin = channels_in
        self.cm = channels_mem

        self.convP = nn.Conv2d(channels_in, channels_mem, kernel_size=1, padding=0, stride=1)
        self.convM = nn.Sequential(
            nn.Conv2d(channels_mem, channels_mem, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            ResBlock(indim=channels_mem),
            nn.Conv2d(channels_mem, channels_mem, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            ResBlock(indim=channels_mem),
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels_mem, 1)

    def forward(self, feat_ref, feat):

        feat_in = torch.cat([feat_ref, feat], dim=1)
        featP = F.relu(self.convP(feat_in))
        featM = self.convM(featP)
        output = torch.sigmoid(self.fc(self.pool(featM).squeeze()))

        return output


class Conv_decouple(nn.Module):

    def __init__(self, inplanes, planes):
        super(Conv_decouple, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class STAN(nn.Module):

    def __init__(self, opt):
        super(STAN, self).__init__()

        keydim = opt.keydim
        valdim = opt.valdim
        arch = opt.arch

        expand = CHANNEL_EXPAND[arch]

        self.Encoder_M = Encoder_M(arch) 
        self.Encoder_Q = Encoder_Q(arch)

        self.keydim = keydim
        self.valdim = valdim

        self.KV_M_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_m4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)

        self.Memory1 = Memory()
        self.Memory2 = Memory()
        self.Memory3 = Memory()
        self.Memory4 = Memory()
        self.SpatialMemory = Memory()
        self.conv_decouple = Conv_decouple(2048, 1024)
        self.Decoder = Decoder(2*valdim, 256, expand)

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects): 
        frame_batch = []
        mask_batch = []
        bg_batch = []
        for o in range(1, num_objects+1): # 1 - no
            frame_batch.append(frame)
            mask_batch.append(masks[:, o])

        for o in range(1, num_objects+1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))

        frame_batch = torch.cat(frame_batch, dim=0)
        mask_batch = torch.cat(mask_batch, dim=0)
        bg_batch = torch.cat(bg_batch, dim=0)

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch) # no, c, h, w
        _, c, h, w = r4.size()
        memfeat = r4
        k4, v4 = self.KV_M_r4(memfeat)
        k4 = k4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim)
        v4 = v4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)
        
        return k4, v4, r4

    def segment(self, frame, keys, values, num_objects, max_obj, opt, frame_idx, keys_dict, vals_dict, patch=2, is_testing=False):
        r4, r3, r2, _ = self.Encoder_Q(frame)
        n, c, h, w = r4.size()
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)

        if is_testing and opt.adapt_memory and (frame_idx >= opt.adapt_memory_maxsize):  # frame_idx=5为第6张！！！！！！！
            adapt_keys, adapt_vals = [], []
            score_dict = {}
            key_preframe = keys_dict[frame_idx-1]
            val_preframe = vals_dict[frame_idx-1]
            curr_key = k4.reshape(k4.shape[0], k4.shape[1], -1).permute(0, 2, 1)
            for idx in range(keys.shape[0]):
                score_dict[idx] = torch.cosine_similarity(curr_key, keys[idx: idx+1], dim=1).squeeze().mean()
            score_dict[keys.shape[0]] = torch.cosine_similarity(curr_key, key_preframe, dim=1).squeeze().mean()
            sorted_idx_list = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)[: opt.adapt_memory_maxsize]
            for idx, _ in sorted_idx_list:
                if idx==keys.shape[0]:
                    adapt_keys.append(key_preframe)
                else:
                    adapt_keys.append(keys[idx: idx+1])
                if idx==keys.shape[0]:
                    adapt_vals.append(val_preframe)
                else:
                    adapt_vals.append(values[idx: idx+1])

            keys = torch.cat(adapt_keys, dim=0)
            values = torch.cat(adapt_vals, dim=0)
            adapt_keys.clear()
            adapt_vals.clear()
        
        m4 = torch.zeros_like(r4)
        BT_ks, HW_ks, C_ks = keys.size()  # BT, HW, C --> BT, H, W, C
        keys = keys.reshape(BT_ks, int(math.sqrt(HW_ks)), int(math.sqrt(HW_ks)), C_ks)
        BT_ks, H_ks, W_ks, C_ks = keys.size()
        BT_vs, HW_vs, C_vs = values.size()  # BT, HW, C --> BT, H, W, C
        values = values.reshape(BT_vs, int(math.sqrt(HW_vs)), int(math.sqrt(HW_vs)), C_vs)
        BT_vs, H_vs, W_vs, C_vs = values.size()
        assert H_ks == H_vs and W_ks == W_vs
        cut_H = H_ks // patch
        cut_W = W_ks // patch
        keys_patch, values_patch = keys[:, 0:cut_H, 0:cut_W, :], values[:, 0:cut_H, 0:cut_W, :]  # BT,Hp,Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0), cut_H*cut_W, keys_patch.size(3))  # BT,Hp*Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0)*cut_H*cut_W, keys_patch.size(2))  # BT*Hp*Wp,C
        keys_patch = keys_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0), cut_H*cut_W, values_patch.size(3))  # BT,Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0) * cut_H * cut_W, values_patch.size(2))  # BT*Hp*Wp,C
        values_patch = values_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        k4e_patch, v4e_patch = k4e[:, :, 0:cut_H, 0:cut_W], v4e[:, :, 0:cut_H, 0:cut_W]  # B,C,Hp,Wp
        m4_patch, _ = self.Memory1(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4[:, :, 0:cut_H, 0:cut_W] = m4_patch
        keys_patch, values_patch = keys[:, 0:cut_H, cut_W:, :], values[:, 0:cut_H, cut_W:, :]  # BT,Hp,Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0), cut_H * (W_ks-cut_W), keys_patch.size(3))  # BT,Hp*Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0) * cut_H * (W_ks-cut_W), keys_patch.size(2))  # BT*Hp*Wp,C
        keys_patch = keys_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0), cut_H * (W_vs-cut_W), values_patch.size(3))  # BT,Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0) * cut_H * (W_vs-cut_W), values_patch.size(2))  # BT*Hp*Wp,C
        values_patch = values_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        k4e_patch, v4e_patch = k4e[:, :, 0:cut_H, cut_W:], v4e[:, :, 0:cut_H, cut_W:]  # B,C,Hp,Wp
        m4_patch, _ = self.Memory2(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4[:, :, 0:cut_H, cut_W:] = m4_patch
        keys_patch, values_patch = keys[:, cut_H:, 0:cut_W, :], values[:, cut_H:, 0:cut_W, :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), (H_ks-cut_H) * cut_W, keys_patch.size(3))  # BT,Hp*Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0) * (H_ks-cut_H) * cut_W, keys_patch.size(2))  # BT*Hp*Wp,C
        keys_patch = keys_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0), (H_vs-cut_H) * cut_W, values_patch.size(3))  # BT,Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0) * (H_vs-cut_H) * cut_W, values_patch.size(2))  # BT*Hp*Wp,C
        values_patch = values_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        k4e_patch, v4e_patch = k4e[:, :, cut_H:, 0:cut_W], v4e[:, :, cut_H:, 0:cut_W]  # B,C,Hp,Wp
        m4_patch, _ = self.Memory3(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4[:, :, cut_H:, 0:cut_W] = m4_patch
        keys_patch, values_patch = keys[:, cut_H:, cut_W:, :], values[:, cut_H:, cut_W:, :]
        keys_patch = keys_patch.reshape(keys_patch.size(0), (H_ks-cut_H) * (W_ks-cut_W), keys_patch.size(3))  # BT,Hp*Wp,C
        keys_patch = keys_patch.reshape(keys_patch.size(0) * (H_ks-cut_H) * (W_ks-cut_W), keys_patch.size(2))  # BT*Hp*Wp,C
        keys_patch = keys_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0), (H_vs-cut_H) * (W_vs-cut_W), values_patch.size(3))  # BT,Hp*Wp,C
        values_patch = values_patch.reshape(values_patch.size(0) * (H_vs-cut_H) * (W_vs-cut_W), values_patch.size(2))  # BT*Hp*Wp,C
        values_patch = values_patch.unsqueeze(0)  # B,T*Hp*Wp,C
        k4e_patch, v4e_patch = k4e[:, :, cut_H:, cut_W:], v4e[:, :, cut_H:, cut_W:]  # B,C,Hp,Wp
        m4_patch, _ = self.Memory4(keys_patch, values_patch, k4e_patch, v4e_patch)
        m4[:, :, cut_H:, cut_W:] = m4_patch
        spatial_keys = keys[keys.size(0) - 1, :, :]
        spatial_keys = spatial_keys.reshape(spatial_keys.size(0)*spatial_keys.size(1), -1)
        spatial_keys = spatial_keys.unsqueeze(0)
        spatial_values = values[values.size(0) - 1, :, :]
        spatial_values = spatial_values.reshape(spatial_values.size(0) * spatial_values.size(1), -1)
        spatial_values = spatial_values.unsqueeze(0)
        km4, vm4 = self.KV_m4(r4)
        m4_spatial, _ = self.SpatialMemory(spatial_keys, spatial_values, km4, vm4)
        m4 = torch.cat((m4, m4_spatial), dim=1)
        m4 = self.conv_decouple(m4)
        logit = self.Decoder(m4, r3e, r2e, frame)
        ps = F.softmax(logit, dim=1)[:, 1]  # no, h, w
        logit = Soft_aggregation(ps, max_obj)  # 1, K, H, W
        
        return logit, ps

    def forward(self, frame, mask=None, keys=None, values=None, num_objects=None, max_obj=None,
                opt=None, frame_idx=None, keys_dict=None, vals_dict=None, patch=2, is_testing=False):

        if mask is not None:  # keys
            return self.memorize(frame, mask, num_objects)
        else:
            return self.segment(frame, keys, values, num_objects, max_obj, opt, frame_idx, keys_dict, vals_dict, patch, is_testing)
