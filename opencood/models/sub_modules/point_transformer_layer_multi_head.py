# I am trying to replace the pillar_vfe layer with a point transformer layer that can output the data similar to PillarVFE.

# I will start by creating a new class PointTransformerVFE that will replace the PillarVFE class in the PointPillarTransformer class.

import torch
import torch.nn as nn
import torch.nn.functional as F
import opencood.libs.pointops as pointops
import einops
from .utils import LayerNorm1d
from opencood.models.sub_modules.point_transformer_seg import TransitionDown
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from torch.utils.checkpoint import checkpoint

class PointTransformerLayer_MH(nn.Module):
    def __init__(self, in_planes, out_planes, no_of_heads, share_planes=8, nsample=16):
        super().__init__()
        self.no_of_heads = no_of_heads
        assert out_planes % no_of_heads == 0 # Here out_planes must be divisible by no_of_heads
        self.mid_planes = mid_planes = out_planes // no_of_heads
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            nn.LayerNorm(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            nn.LayerNorm(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        """
        Forward pass for the PointTransformerLayer.

        Args:
            pxo (tuple): A tuple containing:
                - p (torch.Tensor): Tensor of shape (n, 3) representing the point coordinates.
                - x (torch.Tensor): Tensor of shape (n, c) representing the point features.
                - o (torch.Tensor): Tensor of shape (b) representing the batch offsets.

        Returns:
            torch.Tensor: Tensor of shape (n, out_planes) representing the transformed point features.
        """
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        # Reshape to multi-head format: (n, no_of_heads, mid_planes)


        x_q = x_q.view(-1, self.no_of_heads, self.mid_planes)
        x_k = x_k.view(-1, self.no_of_heads, self.mid_planes)
        x_v = x_v.view(-1, self.no_of_heads, self.mid_planes)
        
        x_k_grouped, idx = pointops.knn_query_and_group(
            x_k.view(-1, self.out_planes),
            p, o,
            new_xyz=p,
            new_offset=o,
            nsample=self.nsample,
            with_xyz=True
        )
        # x_k_grouped: (n, nsample, 3 + out_planes) = (n, nsample, 3 + no_of_heads * mid_planes)
        x_v_grouped, _ = pointops.knn_query_and_group(
            x_v.view(-1, self.out_planes),
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )

        # x_v: (n, nsample, no_of_heads * mid_planes)
        p_r, x_k_grouped = x_k_grouped[:, :, 0:3], x_k_grouped[:, :, 3:]  # p_r represents the relative position encoding
        p_r = self.linear_p(p_r)

        # Reshape for multihead processing
        p_r = p_r.view(-1, self.nsample, self.no_of_heads, self.mid_planes)
        x_k_grouped = x_k_grouped.view(-1, self.nsample, self.no_of_heads, self.mid_planes)
        x_v_grouped = x_v_grouped.view(-1, self.nsample, self.no_of_heads, self.mid_planes)


        # r_qk = x_k_grouped + p_r - x_q.unsqueeze(1)
        # # Compute the attention weights

        # N, _, H, D = r_qk.shape
        # r_qk_flat = r_qk.reshape(N * self.nsample * self.no_of_heads, D)

        # w = self.linear_w(r_qk_flat)  # (N*nsample*no_of_heads, out_planes//share_planes)
        # w = w.view(N, self.nsample, self.no_of_heads, self.out_planes // self.share_planes)


        # w = w.mean(dim=-1)  # (N, nsample, H)
        # w = self.softmax(w) # softmax over nsample dimension

        # w = w.unsqueeze(-1)
        # x = (x_v_grouped + p_r) * w  # (N, nsample, H, D)
        # x = x.sum(dim=1)  # sum over nsample -> (N, H, D)

        # x = x.view(N, H * D)  # (N, out_planes)

        # return x
        
        def _checkpointed_attention(x_q_cp, x_k_cp, x_v_cp, p_r_cp):
            r_qk = x_k_cp + p_r_cp - x_q_cp.unsqueeze(1)
            N, _, H, D = r_qk.shape
            r_qk_flat = r_qk.reshape(N * self.nsample * self.no_of_heads, D)
            w = self.linear_w(r_qk_flat)
            w = w.view(N, self.nsample, self.no_of_heads, self.out_planes // self.share_planes)
            w = w.mean(dim=-1)
            w = self.softmax(w)
            w = w.unsqueeze(-1)
            x_out = (x_v_cp + p_r_cp) * w
            x_out = x_out.sum(dim=1)
            x_out = x_out.view(N, H * D)
            return x_out
    
        if torch.is_grad_enabled():
            x_out = checkpoint(_checkpointed_attention, x_q, x_k_grouped, x_v_grouped, p_r)
        else:
            x_out = _checkpointed_attention(x_q, x_k_grouped, x_v_grouped, p_r)

        return x_out

class Bottleneck_MH(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, no_of_heads, share_planes=8, nsample=16):
        super(Bottleneck_MH, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer_MH(planes, planes, no_of_heads, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]

class PointTransformerVFE_MH(nn.Module): 
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, pillar_vfe_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_point_features = num_point_features
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_threshold = self.model_cfg['sparse_threshold']
        self.pillar_vfe = PillarVFE(pillar_vfe_cfg,
                                    num_point_features=num_point_features,
                                    voxel_size=voxel_size,
                                    point_cloud_range=point_cloud_range)

        self.num_filters = self.model_cfg['num_filters']
        self.num_encoders = self.model_cfg['num_encoders']
        self.strides = self.model_cfg['strides']
        self.nsamples = self.model_cfg['nsamples']
        self.out_channels = self.model_cfg['out_channels']
        self.no_of_heads = self.model_cfg['no_of_heads']
        self.in_layers = num_point_features - 3

        assert self.num_encoders == len(self.strides) == len(self.nsamples)

        for i in range(self.num_encoders):
            setattr(self, f'enc{i}', self.make_enc(Bottleneck_MH, 
                                                   out_planes=self.out_channels[i], 
                                                   blocks=1, 
                                                   share_planes=8,
                                                   no_of_heads=self.no_of_heads, 
                                                   stride=self.strides[i], 
                                                   nsample=self.nsamples[i])
                    )

    

    def make_enc(self, block, out_planes, blocks, share_planes, no_of_heads, stride=1, nsample=16):
        layers = [
            TransitionDown(
                in_planes=self.in_layers,
                out_planes=out_planes,
                stride=stride,
                nsample=nsample
            )
        ]
        self.in_layers = out_planes
        for _ in range(blocks):
            layers.append(
                block(
                    in_planes=self.in_layers,
                    planes=out_planes,
                    no_of_heads=no_of_heads,
                    share_planes=share_planes,
                    nsample=nsample
                )
            )
        return nn.Sequential(*layers)
        

    def forward(self, batch_dict): 
        sparse_threshold = self.sparse_threshold

        # create mask for dense and sparse voxels
        mask = batch_dict['voxel_num_points'].clone().detach() > sparse_threshold

        # split batch_dict into dense and sparse
        dense_dict = {
            'voxel_features': batch_dict['voxel_features'][mask],
            'voxel_num_points': batch_dict['voxel_num_points'][mask],
            'voxel_coords': batch_dict['voxel_coords'][mask]
        }

        sparse_dict = {
            'voxel_features': batch_dict['voxel_features'][~mask],
            'voxel_num_points': batch_dict['voxel_num_points'][~mask],
            'voxel_coords': batch_dict['voxel_coords'][~mask]
        }

        # process dense and sparse voxels
        dense_pillar_features = self.process_dense(dense_dict) if dense_dict['voxel_features'].size(0) > 0 else None
        sparse_pillar_features = self.process_sparse(sparse_dict) if sparse_dict['voxel_features'].size(0) > 0 else None

        # initialize pillar_features tensor
        pillar_features_shape = (len(batch_dict['voxel_num_points']), 
                                dense_pillar_features.shape[1] if dense_pillar_features is not None else sparse_pillar_features.shape[1])
        pillar_features = torch.zeros(pillar_features_shape, device=dense_pillar_features.device if dense_pillar_features is not None else sparse_pillar_features.device)

        # combine dense and sparse pillar features
        if dense_pillar_features is not None:
            dense_pillar_features = dense_pillar_features.to(pillar_features.dtype)
            pillar_features = pillar_features.masked_scatter(mask.unsqueeze(-1), dense_pillar_features)
        if sparse_pillar_features is not None:
            sparse_pillar_features = sparse_pillar_features.to(pillar_features.dtype)
            pillar_features = pillar_features.masked_scatter(~mask.unsqueeze(-1), sparse_pillar_features)

        batch_dict['pillar_features'] = pillar_features

        return batch_dict
    

    def process_dense(self, batch_dict):
        # input: batch_dict: voxel_features, voxel_num_points, voxel_coords
        # output: batch_dict['pillar_features']: (batch_size, num_features)
        # We should align output of this function with the output of PillarVFE
        # as (batch_size, num_voxels, num_features) right?
        
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        points = []

        for i in range(voxel_features.shape[0]):
            points.append(voxel_features[i, :voxel_num_points[i], :])
        
        points = torch.cat(points).to(voxel_features.device)

        p_e = points[:, :3].contiguous() # should be absolute_xyz in voxel_features
        x_e = points[..., 3:].contiguous() # (batch_size, num_voxels, num_features)
        o_e = torch.cumsum(voxel_num_points, dim=0).int().contiguous()

        for i in range(self.num_encoders):
            enc_layer = getattr(self, f'enc{i}')
            p_e, x_e, o_e = enc_layer([p_e, x_e, o_e])
        
        x = []
        for i in range(o_e.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o_e[0], o_e[0]
            else:
                s_i, e_i, cnt = o_e[i - 1], o_e[i], o_e[i] - o_e[i - 1]
            x_b = x_e[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0) # (batch_size, num_features)
        # x = self.cls(x) # Do we need to put classifier here?
        return x

    def process_sparse(self, batch_dict):
        # input: batch_dict: voxel_features, voxel_num_points, voxel_coords
        # output: batch_dict['pillar_features']: (batch_size, num_voxels, num_features)
        batch_dict = self.pillar_vfe(batch_dict)
        x = batch_dict['pillar_features']
        return x