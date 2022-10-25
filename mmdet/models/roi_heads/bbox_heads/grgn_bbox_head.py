# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import math

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from mmcv.cnn import build_norm_layer
from .bbox_head import BBoxHead

# [K,4] -> [K,K,4] # get the pairwise box geometric feature
def geometric_encoding_single(boxes):
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
    w = x2 - x1
    h = y2 - y1
    center_x = 0.5 * (x1+x2)
    center_y = 0.5 * (y1+y2)

    # [K,K]
    delta_x = center_x - torch.transpose(center_x,0,1)
    delta_x = delta_x / w
    delta_x = torch.log(torch.abs(delta_x).clamp(1e-3))

    delta_y = center_y - torch.transpose(center_y,0,1)
    delta_y = delta_y / w
    delta_y = torch.log(torch.abs(delta_y).clamp(1e-3))

    delta_w = torch.log(w / torch.transpose(w,0,1))

    delta_h = torch.log(h / torch.transpose(h,0,1))

    # [K,K,4]
    output = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=2)

    return output

def geometric_encoding_batch(boxes, bs):
    '''
    boxes: tensor[bs*roi_num,4]
    bs: batch size
    '''

    boxes = boxes.reshape(bs,-1,4)
    bbox_encoding_batch = []
    for bs_id in range(0,bs):
        bbox_encoding = geometric_encoding_single(boxes[bs_id].squeeze())
        bbox_encoding_batch.append(bbox_encoding)
    # [bs,roi_num,roi_num,4]
    output = torch.stack(bbox_encoding_batch, dim=0)

    return output

class RelationModule(nn.Module):
    """
    Simple layer for self-attention, 

    Input:
    group: the head number of relation module
    geo_feat_dim: embedding the position

    Output:
    [bs*roi_num,1024]
    """

    def __init__(self, box_feat_dim=1024, group=16, geo_feat_dim=64):
        super(RelationModule, self).__init__()
        self.box_feat_dim = box_feat_dim
        self.group = group
        self.geo_feat_dim = geo_feat_dim
        self.group_feat_dim = int(box_feat_dim / group) # 1024/16=64

        self.tanh = nn.Tanh()
        self.geo_emb_fc = nn.Linear(4, geo_feat_dim) # 4->64
        self.box_geo_conv = ConvModule(geo_feat_dim, group, 1) # 1x1 conv
        self.query_fc = nn.Linear(box_feat_dim, box_feat_dim) 
        self.key_fc = nn.Linear(box_feat_dim, box_feat_dim)
        self.group_conv = ConvModule(group*box_feat_dim, box_feat_dim, 1, groups=group) # 16*1024->1024


    def forward(self, box_appearance_feat, boxes, bs):
        roi_num = int(box_appearance_feat.size(0)/bs)

        # [K,4] -> [bs,roi_num,roi_num,4]
        # given the absolute box, get the pairwise relative geometric coordinates
        box_geo_encoded = geometric_encoding_batch(boxes,bs)
        # [bs,roi_num,roi_num,4] -> [bs,roi_num,roi_num,64]
        box_geo_feat = self.tanh(self.geo_emb_fc(box_geo_encoded))
        
        # [bs,64,roi_num,roi_num]
        box_geo_feat = box_geo_feat.permute(0,3,1,2)  

        # [bs,16,roi_num,roi_num]
        box_geo_feat_wg = self.box_geo_conv(box_geo_feat)
        # [bs,roi_num,16,roi_num]
        box_geo_feat_wg = box_geo_feat_wg.permute(0,2,1,3)

        # now we get the appearance stuff
        # [bs,roi_num,1024]
        box_appearance_feat = box_appearance_feat.reshape(bs,-1,self.box_feat_dim)
        query = self.query_fc(box_appearance_feat)
        # split head
        # [bs,roi_num,16,1024/16]
        query = query.reshape(bs, -1, self.group, self.group_feat_dim)
        query = query.permute(0, 2, 1, 3) # [bs,16,roi_num,1024/16]

        key = self.key_fc(box_appearance_feat)
        # split head
        # [bs,roi_num,16,1024/16]
        key = key.reshape(bs, -1, self.group, self.group_feat_dim)
        key = key.permute(0, 2, 1, 3)  # [bs,16,roi_num,1024/16]

        value = box_appearance_feat

        key = key.permute(0, 1, 3, 2)  # [bs,16,1024/16,roi_num]
        # [bs,16,roi_num,1024/16]*[bs,16,1024/16,roi_num] ->[bs,16,roi_num,roi_num]
        logits = torch.matmul(query, key)
        logits_scaled = (1.0 / math.sqrt(self.group_feat_dim)) * logits
        logits_scaled = logits_scaled.permute(0, 2, 1, 3)  # [bs,roi_num,16,roi_num]

        # [bs,roi_num,16,roi_num]
        weighted_logits = torch.log(box_geo_feat_wg.clamp(1e-6)) + logits_scaled
        weighted_softmax = F.softmax(weighted_logits,dim=3)

        # need to reshape for matmul [bs,roi_num*16,roi_num]
        weighted_softmax = weighted_softmax.reshape(bs, roi_num*self.group, roi_num)

        # [bs,roi_num*16,roi_num] * [bs,roi_num,1024] -> [bs,roi_num*16,1024]
        output = torch.matmul(weighted_softmax, value)

        # [bs,roi_num,16*1024,1,1]
        output = output.reshape(bs, -1, self.group*self.box_feat_dim,1,1)

        out_batch = []
        for bs_id in range(0,bs):
            out_single = self.group_conv(output[bs_id].squeeze(0))
            out_batch.append(out_single)

        output = torch.stack(out_batch, dim=0)
        output = output.reshape(bs*roi_num,self.box_feat_dim)

        return output

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907

    implement the function of: Y = A·(X·W)
        input X: [C,D1]
        weight W: [D1,D2]
         --> (X·W): [C,D2]
        adjacency matrix A: [C,C]
        output Y: [C, D2]
    C is the class number, D1 is the embedding dimension, D2 is the latent feature dimension
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # add weight and bias into the model as trainable Parameter  
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches. Support some functions and initialize hyper parameters.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
        cls_gcn_cfg (dict): Hyper parameters of class branch.
        adj_mode: Select type of adjacency matrix. Default 'A_img_lvl_eye'.
            Support 'A_img_lvl', 'A_img_lvl_eye', 'A_obj_lvl', 'A_iou', 'A_iof', 'A_giou'.
            Besides the predefined adjacency matrix, we also provide 'random', 'learn', 'ones'.
        emb_mode: Select type of embeddings of each class, a class can be embeded to a vector
            with dim [1, emb_channel]. Support 'label', 'word', 'random', 'ones'.
        fusion_mode: Select the way to fuse features of roi_feat and enhanced_feat.
            Note that the fusion here refers to the fusion of enhanced gcn feature and 
            x_roi feature. The fusion ways of relation information(gcns) and image meta 
            informaiton (x_feat or position) are fix. Default 'dot'. Support 'dot', 'sum', 'cat'.
        norm: Nomarlize the output of gcns. Default 'GN'. Support 'GN', 'BN', 'LN'
    """  

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 num_gcn_convs=0,
                 num_gcn_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 cls_gcn_cfg:dict=None,
                 reg_gcn_cfg:dict=None,
                 gcn_cfg:dict=None,
                 use_attention=False,
                 att_group=16,
                 *args,
                 **kwargs):
        super(GCNConvFCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.num_gcn_convs = num_gcn_convs
        self.num_gcn_fcs = num_gcn_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # setting of relation module
        self.use_attention = use_attention
        self.att_group = att_group

        # setting of overall gcn 
        self.emb_channel = gcn_cfg['emb_channel']
        self.norm = gcn_cfg['norm']
        self.stop_grad = gcn_cfg['use_stop_grad']
        self.fusion_mode = gcn_cfg['fusion_mode']
        self.emb_mode = gcn_cfg['emb_mode']
        self.num_gcns = gcn_cfg['num']
        self.gcn_bias = gcn_cfg['bias']
        self.gcn_act = gcn_cfg['act']

        # setting of semantic guidance module
        self.cls_gcn = cls_gcn_cfg['use_gcn']
        self.cls_adj_mode = cls_gcn_cfg['adj_mode']

        # setting of postional guidance module
        self.reg_gcn = reg_gcn_cfg['use_gcn']
        self.reg_adj_mode = reg_gcn_cfg['adj_mode']

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        # initialize the adjacency matrix
        if self.cls_adj_mode == 'random' or self.reg_adj_mode == 'random':
            self.cls_A = torch.randn(self.num_classes,self.num_classes)
            self.reg_A = torch.randn(self.num_classes,self.num_classes)
        elif self.cls_adj_mode == 'ones' or self.reg_adj_mode == 'ones':
            self.cls_A = torch.ones(self.num_classes,self.num_classes)
            self.reg_A = torch.ones(self.num_classes,self.num_classes)    
        elif self.cls_adj_mode == 'learn' or self.reg_adj_mode == 'learn':
            self.cls_A = Parameter(torch.randn(self.num_classes, self.num_classes))    
            self.reg_A = Parameter(torch.randn(self.num_classes, self.num_classes))   
        else:
            with open(gcn_cfg['adj_path'], 'rb') as f:
                adjacency_info = pickle.load(f)
            self.reg_A = torch.from_numpy(adjacency_info[f'{self.reg_adj_mode}']).float()
            self.cls_A = torch.from_numpy(adjacency_info[f'{self.cls_adj_mode}']).float()

        # load word embeddings
        if self.cls_gcn:
            with open(gcn_cfg['emb_path'], 'rb') as f:
                word_embeddings_info1 = pickle.load(f)
            if self.emb_mode == 'label':
                self.cls_word_embeddings = torch.from_numpy(word_embeddings_info1['emb_label']).float()
            elif self.emb_mode == 'word':
                self.cls_word_embeddings = torch.from_numpy(word_embeddings_info1['emb_word']).float()
            elif self.emb_mode == 'random':
                self.cls_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'ones':
                self.cls_word_embeddings = torch.ones(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'learn':
                self.cls_word_embeddings = Parameter(torch.randn(self.num_classes, self.emb_channel))
                

        if self.reg_gcn:
            with open(gcn_cfg['emb_path'], 'rb') as f:
                word_embeddings_info2 = pickle.load(f)
            if self.emb_mode == 'label':
                self.reg_word_embeddings = torch.from_numpy(word_embeddings_info2['emb_label']).float()
            elif self.emb_mode == 'word':
                self.reg_word_embeddings = torch.from_numpy(word_embeddings_info2['emb_word']).float()
            elif self.emb_mode == 'random':
                self.reg_word_embeddings = torch.randn(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'ones':
                self.reg_word_embeddings = torch.ones(self.num_classes, self.emb_channel)
            elif self.emb_mode == 'learn':
                self.reg_word_embeddings = Parameter(torch.randn(self.num_classes, self.emb_channel))


        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        if self.gcn_act == 'leakyrelu':
            self.gcn_relu = nn.LeakyReLU(0.2)
        elif self.gcn_act == 'relu':
            self.gcn_relu = nn.ReLU()

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs'),
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_gcns(self, num_gcns, in_channels, use_bias=False):
        gcns = nn.ModuleList()
        for i in range(num_gcns):
            in_channels = (in_channels if i == 0 else self.fc_out_channels)
            gcns.append(GraphConvolution(in_channels, self.fc_out_channels, use_bias))

        return gcns


    def bbox_xyxy_to_relative_xyxy(self, bbox, img_metas, roi_bs_ids=[]):
        """Convert bbox coordinates from absolute  (x1, y1, x2, y2) 
            to relative (x1, y1, x2, y2).

        Args:
            bbox (Tensor): Shape (n, 4) for bboxes.
            img_metas (list): meta information of a batch imgs
            roi_bs_ids (Tensor): Shape (n,), img id for each bbox.
        Returns:
            Tensor: Converted bboxes.
        """
        bs = len(img_metas)
        img_shapes = [img_meta['img_shape'] for img_meta in img_metas]
        img_shapes = np.array(img_shapes)[:,:2]
        img_shapes = torch.from_numpy(img_shapes)

        if bbox.size(0) % bs == 0:
            w = img_shapes[:,1].reshape(-1,1)
            h = img_shapes[:,0].reshape(-1,1)
            _mat = torch.cat((w,h,w,h),dim=1).reshape(bs,-1,4).to(bbox.device)
            
            bbox_new = bbox.reshape(bs,-1,4)
            bbox_new = bbox_new / _mat
            bbox_new = bbox_new.reshape(-1,4)
        else:
            print('Warning!rois less than default setting...')
            bs_ids = torch.unique(roi_bs_ids, sorted=True)
            for bs_id in bs_ids:
                inds = (roi_bs_ids == bs_id.item())
                w = img_shapes[bs_id.int(),1].reshape(-1,1)
                h = img_shapes[bs_id.int(),0].reshape(-1,1)
                _mat = torch.cat((w,h,w,h),dim=1).reshape(-1,4).to(bbox.device)
                bbox[inds,:] = bbox[inds,:] / _mat
            bbox_new = bbox

        return bbox_new
    
    def conv1d_w_dim_process(self, conv1d, inp):
        """conv1d with dimension proessing 

        Args:
            conv1d (nnmodule), act (nnmodule)
            inp (tensor): [C, D] 
        Returns:
            out (tensor): [1, D]
        """
        inp = inp.unsqueeze(0).permute(0,2,1)
        out = self.relu(conv1d(inp))
        out = out.permute(0,2,1).squeeze()
        return out

    def fusion(self, x1, x2, mode='sum'):
        """fusion two tensors with same dimension

        Args:
            x1 (tensor): [B*R, D] 
            x2 (tensor): [B*R, D] 
        Returns:
            out (tensor): [B*R, D] or [B*R, 2D]
        """
        if mode == 'sum':
            out = x1 + x2
        elif mode == 'dot':
            out = x1 * x2
        elif mode == 'cat':
            out = torch.cat((x1, x2), dim=-1)
        return out

    def fusion_w_dim_process(self,roi_feat,gcn_feat,roi_bs_ids,mode='sum'):
        """fusion two tensors 

        Args:
            roi_feat (tensor): [B*R, D] 
            gcn_feat (tensor): [B, D] 
            roi_bs_ids: [B*R]
        Returns:
            out (tensor): [B*R, D] or [B*R, 2D]
        """
        bsR = roi_feat.size(0)
        bs = gcn_feat.size(0)
        D = gcn_feat.size(1)
        if bsR % bs == 0:
            gcn_feat = gcn_feat.reshape(bs,1,D)
            roi_feat = roi_feat.reshape(bs,-1,D) # [bs,rois,1024]
            if mode == 'sum':
                roi_feat = roi_feat + gcn_feat # [bs,rois,1024] 
            elif mode == 'dot':
                roi_feat = roi_feat * gcn_feat # [bs,rois,1024] 
            elif mode == 'cat':
                gcn_feat = gcn_feat.reshape(bs,-1,D).repeat(1, int(bsR/bs), 1)
                roi_feat = torch.cat((roi_feat, gcn_feat), dim=-1) # [bs,rois,2048] 
            roi_feat = roi_feat.reshape(bsR,-1) 
        else:
            print('Warning!rois less than default setting...')
            bs_ids = torch.unique(roi_bs_ids, sorted=True)
            for bs_id in bs_ids:
                inds = (roi_bs_ids == bs_id.item())
                if mode == 'sum':
                    roi_feat[inds,:] = roi_feat[inds,:] + gcn_feat[bs_id.int(),:]
                elif mode == 'dot':
                    roi_feat[inds,:] = roi_feat[inds,:] * gcn_feat[bs_id.int(),:]
                elif mode == 'cat':
                    roi_feat[inds,:] = torch.cat((roi_feat[inds,:], gcn_feat[bs_id.int(),:]), dim=-1)

        return roi_feat


@HEADS.register_module()
class GRGNBBoxHead(GCNConvFCBBoxHead):
    '''
        Box head implementation for Graph-based Relation Guiding Network (GRGN).
    '''
    def __init__(self, *args, **kwargs):
        super(GRGNBBoxHead, self).__init__(
            num_shared_fcs=2,
            *args,
            **kwargs)

        # if self.cls_gcn or self.reg_gcn:
        if self.cls_gcn:
            if self.norm == 'GN':
                norm_cfg=dict(type='GN', num_groups=8, requires_grad=True)
                self.gcn_norm = build_norm_layer(norm_cfg, self.fc_out_channels)[1]
            elif self.norm == 'BN':
                self.gcn_norm = nn.BatchNorm1d(self.fc_out_channels)
            elif self.norm == 'LN':
                self.gcn_norm = nn.LayerNorm(self.fc_out_channels)

        if self.cls_gcn:
            # visual embedding
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.map_fcs = nn.ModuleList()
            self.map_fcs.append(nn.Linear(self.in_channels, self.fc_out_channels))
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='map_fcs')])]

            self.gcns_cls = self._add_gcns(self.num_gcns, self.emb_channel, self.gcn_bias)
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='gcns_cls')])]

            self.cls_gcn_conv1d = nn.Conv1d(self.fc_out_channels, self.fc_out_channels, self.num_classes)
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='cls_gcn_conv1d')])]

        if self.reg_gcn:
            # position embedding
            self.pos_emb_fcs = nn.ModuleList()
            self.pos_emb_fcs.append(nn.Linear(4, 64))
            self.pos_emb_fcs.append(nn.Linear(64, self.num_classes))
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='pos_emb_fcs')])]

            self.gcns_reg_enh = self._add_gcns(self.num_gcns, self.emb_channel, self.gcn_bias) 
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='gcns_reg_enh')])]
                
        if self.use_attention:
            self.attention_leyers = nn.ModuleList()
            self.attention_leyers.append(RelationModule(self.fc_out_channels, self.att_group))
            self.attention_leyers.append(RelationModule(self.fc_out_channels, self.att_group))
            self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='attention_leyers')])]

        if self.fusion_mode == 'cat':
            if self.cls_gcn:
                self.cls_fusion_cat_fc = nn.Linear(self.fc_out_channels*2, self.fc_out_channels)
                self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='cls_fusion_cat_fc')])]
            if self.reg_gcn:
                self.reg_fusion_cat_fc = nn.Linear(self.fc_out_channels*2, self.fc_out_channels)
                self.init_cfg += [dict(type='Xavier',distribution='uniform',override=[dict(name='reg_fusion_cat_fc')])]

    def forward(self, x_roi, x_map, rois, img_metas):
        """
        Arguments
        ---------
            x_roi: roi feature. [bs, 256, 7, 7]
            x_map: the whole feature map from backbone. [bs, 2048, H, W]
            rois: the box position. [batch_ind, x1, y1, x2, y2]
            img_metas: provide image information such as 'img_shape'

        """
    
        bs = x_map[-1].size(0)
        roi_num = x_roi.size(0)
        roi_bs_ids = rois[:, 0]
        gpu_device = x_roi.device

        if not (roi_num % bs == 0):
            print('Warning!rois less than default setting...')

        if self.stop_grad:
            c5_feat = x_map[-1].detach() 
            rois = rois[:,1:].detach()
        else:
            c5_feat = x_map[-1]
            rois = rois[:,1:]

        # shared fc1
        x_roi = x_roi.flatten(1)
        x_roi = self.relu(self.shared_fcs[0](x_roi))

        # relation module 1
        if self.use_attention & (roi_num % bs == 0):
            attention_feat = self.attention_leyers[0](x_roi, rois, bs)
            x_roi = self.relu(x_roi + attention_feat)

        # shared fc2
        x_roi = self.relu(self.shared_fcs[1](x_roi))

        # relation module 2
        if self.use_attention & (roi_num % bs == 0):
            attention_feat = self.attention_leyers[1](x_roi, rois, bs)
            x_roi = self.relu(x_roi + attention_feat)

        # separate branches
        x_cls = x_roi
        x_reg = x_roi

        # /*------- generate enhance feature -------*/
        if self.cls_gcn:
            # x_map->pool->fc, output dim [bs,1024]
            c5_feat = self.global_pool(c5_feat)
            c5_feat = c5_feat.flatten(1)
            for fc in self.map_fcs:
                c5_feat = self.relu(fc(c5_feat))

            # emb->gcns->norm->conv1d, output dim [1,1024]
            if self.cls_adj_mode == 'learn':
                adj = self.cls_A.to(gpu_device) 
            else:
                adj = self.cls_A.detach().to(gpu_device) 
            cls_gcn_fea = self.cls_word_embeddings.to(gpu_device) 
            for gcn in self.gcns_cls:
                cls_gcn_fea = self.gcn_relu(gcn(cls_gcn_fea, adj))
            cls_gcn_fea = cls_gcn_fea.reshape(self.num_classes, self.fc_out_channels)
            if self.norm:
                cls_gcn_fea = self.gcn_norm(cls_gcn_fea)

            # [bs,1024] + [1,1024] -> [bs,1024]
            # fuse the features from (input image with visual information) and (class relation)
            cls_gcn_conv1d_feat = self.conv1d_w_dim_process(self.cls_gcn_conv1d, cls_gcn_fea)
            cls_enhance_feat = cls_gcn_conv1d_feat + c5_feat # [bs,1024]
        
        if self.reg_gcn:
            # pos->pos_emb->fc, output dim [roi_num,C]
            rois_emb = self.bbox_xyxy_to_relative_xyxy(rois, img_metas, roi_bs_ids)
            for fc in self.pos_emb_fcs:
                rois_emb = self.relu(fc(rois_emb))

            # emb->gcns->norm, output dim [C,1024]
            if self.reg_adj_mode == 'learn':
                adj = self.reg_A.to(gpu_device) 
            else:
                adj = self.reg_A.detach().to(gpu_device) 
            reg_gcn_enh_fea = self.reg_word_embeddings.to(gpu_device) 
            for gcn in self.gcns_reg_enh:
                reg_gcn_enh_fea = self.gcn_relu(gcn(reg_gcn_enh_fea, adj)) # [C,1024]
            reg_gcn_enh_fea = reg_gcn_enh_fea.reshape(self.num_classes, self.fc_out_channels)
            # if self.norm:
            #     reg_gcn_enh_fea = self.gcn_norm(reg_gcn_enh_fea)

            # [roi_num,C] x [C,1024] -> [roi_num,1024]
            # fuse the features from (input image with sptial information) and (class relation)
            reg_enhance_feat = torch.matmul(rois_emb, reg_gcn_enh_fea) # [roi_num,1024]

        # /*------- fuse enhance feature -------*/
        if self.cls_gcn:
            x_cls = self.fusion_w_dim_process(x_cls, cls_enhance_feat, roi_bs_ids, self.fusion_mode)     
            if self.fusion_mode == 'cat':
                x_cls = self.relu(self.cls_fusion_cat_fc(x_cls))

        if self.reg_gcn:
            x_reg = self.fusion(x_reg, reg_enhance_feat, self.fusion_mode)
            if self.fusion_mode == 'cat':
                x_reg = self.relu(self.reg_fusion_cat_fc(x_reg))

        # classifier
        cls_score = self.fc_cls(x_cls) if self.with_cls else None

        # regressor
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred

