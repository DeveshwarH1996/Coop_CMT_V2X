import torch
import torch.nn as nn

from opencood.models.sub_modules.point_transformer_layer import PointTransformerVFE
from opencood.models.sub_modules.point_transformer_layer_multi_head import PointTransformerVFE_MH
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.sub_modules.pillar_cross_attention import PillarCrossAttention
from torch.utils.checkpoint import checkpoint

class PointPillarPointTransformer(nn.Module):
    def __init__(self, args):
        super(PointPillarPointTransformer, self).__init__()

        self.max_cav = args['max_cav']
        
        # PIllar VFE
        # Need to replace PillarVFE with a version of point transformer that can output the data similar to PillarVFE

        if args.get('point_transformer_vfe'):
            self.point_transformer_vfe = PointTransformerVFE(args['point_transformer_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'],
                                    pillar_vfe_cfg=args['pillar_vfe'])
        elif args.get('point_transformer_vfe_mh'):
            self.point_transformer_vfe = PointTransformerVFE_MH(args['point_transformer_vfe_mh'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'],
                                    pillar_vfe_cfg=args['pillar_vfe'])
        
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        num_features = args['point_pillar_scatter']['num_features']

        if args.get('pillar_cross_attention'):
            self.pillar_cross_attention = PillarCrossAttention(args['pillar_cross_attention'], num_features)
            num_features += args['pillar_cross_attention']['num_output_CA_features']
        
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], num_features)
        
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        if 'core_method' in args['transformer'] and args['transformer']['core_method'] == 'video_codec':
            from opencood.models.sub_modules.video_codec_fusion import VideoCodecFusion
            self.fusion_net = VideoCodecFusion(args['transformer'])
        else:
            self.fusion_net = V2XTransformer(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.point_transformer_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        #TODO: Adjusting voxel_feature structure

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)
        
    

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        
        # n, 4 -> n, c
        batch_dict = self.point_transformer_vfe(batch_dict)
        
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)

        if hasattr(self, 'pillar_cross_attention'):
            batch_dict = self.pillar_cross_attention(batch_dict)
        
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        if 'object_ids' in data_dict:
            object_ids = data_dict['object_ids']
        else:
            object_ids = None

        # transformer fusion
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix, object_ids=object_ids)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict











        # def run_pt_vfe(bd):
        #     return self.point_transformer_vfe(bd)

        # if torch.is_grad_enabled():
        #     batch_dict = checkpoint(run_pt_vfe, batch_dict)
        # else:
        #     batch_dict = run_pt_vfe(batch_dict)

        # # 包裹 scatter
        # def run_scatter(bd):
        #     return self.scatter(bd)
        # if torch.is_grad_enabled():
        #     batch_dict = checkpoint(run_scatter, batch_dict)
        # else:
        #     batch_dict = run_scatter(batch_dict)

        # # pillar cross attention
        # if hasattr(self, 'pillar_cross_attention'):
        #     # def run_cross_attention(bd):
        #     #     return self.pillar_cross_attention(bd)
        #     # if torch.is_grad_enabled():
        #     #     batch_dict = checkpoint(run_cross_attention, batch_dict)
        #     # else:
        #     #     batch_dict = run_cross_attention(batch_dict)
        #     batch_dict = self.pillar_cross_attention(batch_dict)

        # # backbone
        # def run_backbone(bd):
        #     return self.backbone(bd)
        # if  torch.is_grad_enabled():
        #     batch_dict = checkpoint(run_backbone, batch_dict)
        # else:
        #     batch_dict = run_backbone(batch_dict)

        # spatial_features_2d = batch_dict['spatial_features_2d']

        # if self.shrink_flag:
        #     def run_shrink(feat):
        #         return self.shrink_conv(feat)
        #     if torch.is_grad_enabled():
        #         spatial_features_2d = checkpoint(run_shrink, spatial_features_2d)
        #     else:
        #         spatial_features_2d = run_shrink(spatial_features_2d)


        # if self.compression:
        #     def run_compressor(feat):
        #         return self.naive_compressor(feat)
        #     if torch.is_grad_enabled():
        #         spatial_features_2d = checkpoint(run_compressor, spatial_features_2d)
        #     else:    
        #         spatial_features_2d = run_compressor(spatial_features_2d)

        # regroup_feature, mask = regroup(spatial_features_2d, record_len, self.max_cav)
        # prior_encoding = prior_encoding.repeat(
        #     1, 1, 1, 
        #     regroup_feature.shape[3], 
        #     regroup_feature.shape[4]
        # )
        # regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        # regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)

        # def run_fusion(rf, m, scm):
        #     return self.fusion_net(rf, m, scm)

        # if torch.is_grad_enabled():
        #     fused_feature = checkpoint(run_fusion, regroup_feature, mask, spatial_correction_matrix)
        # else:
        #     fused_feature = run_fusion(regroup_feature, mask, spatial_correction_matrix)
        # fused_feature = fused_feature.permute(0, 3, 1, 2)

        # psm = self.cls_head(fused_feature)
        # rm = self.reg_head(fused_feature)
        # output_dict = {'psm': psm, 'rm': rm}

        # return output_dict
