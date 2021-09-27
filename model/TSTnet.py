import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import SAGEConv
from OT_torch_ import cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
import math
from torch_geometric.data import Data
from train import DEVICE
import numpy as np

def getGraphdata(source_share, bs, target_share, target=True):
    segments = torch.reshape(torch.tensor(range(bs)),(-1,int(math.sqrt(bs))))
    src_edge = torch.tensor(getEdge(source_share, segments)).t().contiguous()
    source_share_graph = Data(x=source_share,edge_index=src_edge).to(DEVICE)
    if target == True:
        tar_edge = torch.tensor(getEdge(target_share, segments)).t().contiguous()
        target_share_graph = Data(x=target_share,edge_index=tar_edge).to(DEVICE)
    else:
        target_share_graph =  0
    return source_share_graph, target_share_graph

def getEdge(image, segments, compactness=300, sigma=3.):
    coo = set()
    dire = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    for i in range(1, segments.shape[0]):
        for j in range(1, segments.shape[1]):
            for dx, dy in dire:
                if -1 < i + dx < segments.shape[0] and \
                        -1 < j + dy < segments.shape[1] and \
                        segments[i, j] != segments[i + dx, j + dy]:
                    coo.add((segments[i, j], segments[i + dx, j + dy]))

    coo = np.asarray(list(coo))
    return coo

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels,num_classes,dropout=0.5):
        super(Topology_Extraction, self).__init__()
        self.conv1 = SAGEConv(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SAGEConv(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(32, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes, bias=True)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x_temp_1 = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x_temp_2 = x
        x = self.mlp_classifier(x)
        return F.softmax(x, dim=1), x, x_temp_1, x_temp_2

class vgg16(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, init_weights=True, batch_norm=True):
        super(vgg16, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [32, 32, 64, 64],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        layers = []
        for v in cfg['D']:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, stride=1, kernel_size=3)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(self._get_final_flattened_size(), 4096),
            nn.ReLU(True),
            nn.Dropout())
        self.classifier1 = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.fc = nn.Linear(256, num_classes)
        if init_weights:
            self._initialize_weights()
        
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels,
                             self.patch_size, self.patch_size))
            x = self.features(x)
            t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, test=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x_share = x
        x = self.classifier(x)
        x = self.classifier1(x)
        x = self.fc(x)
        return x, x_share

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Feature_Extractor(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, **kwargs):
        super(Feature_Extractor, self).__init__()
        self.basemodel = vgg16(in_channels,num_classes, patch_size)
        self.classes = num_classes
        self.gcn = Global_graph(1024,num_classes) #patch_size=12, 1024 / patch_size=13, 1600
        self.mmd = MMD_loss(kernel_type='linear')

    def forward(self, source, target=None):
        out = self.basemodel(source)
        src_pred, source_share = out[0], out[1]
        if self.training == True:
            out = self.basemodel(target)
            tar_pred, target_share = out[0], out[1]
            loss = self.mmd(source_share, target_share)
            bs = src_pred.shape[0]
            source_share_graph, target_share_graph = getGraphdata(source_share, bs, target_share)
            src_gcn_pred, _, TST_wd, TST_gwd = self.gcn(source_share_graph,target_share_graph)
        else:
            loss, TST_wd, TST_gwd, tar_pred, src_gcn_pred = 0, 0, 0, 0, 0
        return src_pred, loss, TST_wd, TST_gwd, tar_pred, src_gcn_pred


class Global_graph(nn.Module):
    def __init__(self, in_channels, num_classes=9):
        super(Global_graph, self).__init__()
        self.sharedNet_src = Topology_Extraction(in_channels,num_classes)
        self.sharedNet_tar = Topology_Extraction(in_channels,num_classes)

    def forward(self, source, target):
        wd_ori, gwd_ori = OT(source, target,ori=True)
        out = self.sharedNet_src(source)
        p_source, source, source_share_1, source_share_2 = out[0], out[1], out[2], out[3]
        out = self.sharedNet_tar(target)
        p_target, target, target_share_1, target_share_2 = out[0], out[1], out[2], out[3]
        wd_1, gwd_1 = OT(source_share_1, target_share_1)
        wd_2, gwd_2 = OT(source_share_2, target_share_2)
        wd = wd_ori + wd_1 + wd_2
        gwd = gwd_ori + gwd_1 + gwd_2
        return source, target, wd, gwd

def OT(source_share, target_share, ori=False):
    if ori == True:
        source_share = source_share.x.unsqueeze(0).transpose(2,1)
        target_share = target_share.x.unsqueeze(0).transpose(2,1)
    else:
        source_share = source_share.unsqueeze(0).transpose(2,1)
        target_share = target_share.unsqueeze(0).transpose(2,1)

    cos_distance = cost_matrix_batch_torch(source_share, target_share)
    cos_distance = cos_distance.transpose(1,2)
    # TODO: GW and Gwd as graph alignment loss
    beta = 0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)
    wd = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2), target_share.size(2), iteration=30)
    gwd = GW_distance_uniform(source_share, target_share)
    return torch.mean(wd), torch.mean(gwd)

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return loss
