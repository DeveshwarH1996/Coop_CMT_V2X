import torch
from opencood.models.sub_modules.coop_cmt.resnet import ResNet
from opencood.models.sub_modules.coop_cmt.cp_fpn import CPFPN
from opencood.models.sub_modules.coop_cmt.spconv_voxelize import SPConvVoxelization
from opencood.models.sub_modules.coop_cmt.voxel_encoder import HardSimpleVFE
from opencood.models.sub_modules.coop_cmt.sparse_encoder import SparseEncoder
from opencood.models.sub_modules.coop_cmt.second import SECOND
from opencood.models.sub_modules.coop_cmt.second_fpn import SECONDFPN
from opencood.models.sub_modules.coop_cmt.cmt_head import CMTHead

class CoopCMT(torch.nn.Module):
    
    def __init__(self, args):
        super(CoopCMT, self).__init__()
        self.args = args
        self.img_backbone = ResNet(**args['img_backbone'])
        self.img_neck = CPFPN(**args['img_neck'])
        self.pts_voxel_layer = SPConvVoxelization(**args['pts_voxel_layer'])
        self.pts_voxel_encoder = HardSimpleVFE(**args['pts_voxel_encoder'])
        self.pts_middle_encoder = SparseEncoder(**args['pts_middle_encoder'])
        self.pts_backbone = SECOND(**args['pts_backbone'])
        self.pts_neck = SECONDFPN()
        self.pts_bbox_head = CMTHead(**args['pts_bbox_head'])

        self.loss_cls = FocalLoss()
        self.loss_bbox = L1Loss()
        self.heatmap = GaussianFocalLoss()

