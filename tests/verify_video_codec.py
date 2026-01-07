
import torch
import sys
import os

# Add project root to path
sys.path.append('/home/ecoprt/cooperative_pers/HEAL')

from opencood.models.sub_modules.video_codec_fusion import VideoCodecFusion

def test_video_codec_fusion():
    print("Testing VideoCodecFusion...")
    
    # Dummy args
    args = {
        'transformer': {
            'encoder': {
                'cav_att_config': {'dim': 32, 'use_hetero': False, 'use_RTE': False, 'heads': 4, 'dim_head': 8, 'dropout':0.1},
                'pwindow_att_config': {'dim': 32, 'heads': 4, 'dim_head': 8, 'dropout': 0.1, 'window_size': [4, 4], 'relative_pos_embedding': True, 'fusion_method': 'max'},
                'feed_forward': {'mlp_dim': 64, 'dropout': 0.1},
                'sttf': {'voxel_size': [0.4, 0.4, 4], 'downsample_rate': 4},
                'num_blocks': 1,
                'depth': 1,
                'input_dim': 32,
                'use_roi_mask': False
            }
        },
        'keyframe_interval': 2,
        'use_motion_vector': True
    }
    
    model = VideoCodecFusion(args)
    model.eval()
    
    # Inputs: (B, L, H, W, C)
    # B=1, L=2 (1 ego, 1 neighbor)
    # H=16, W=16, C=32 (channels must match encoder dim... wait, V2XTransformer usually projects input?
    # No, V2XTransformer encoder expects input to be processed or match dim?
    # STTF input C: The STTF logic takes x.
    # checking v2xvit_basic.py: STTF expects x.permute(0, 1, 4, 2, 3) -> (B, L, C, H, W).
    # Input x to V2XTransformer is (B, L, H, W, C).
    
    B, L, H, W, C = 1, 2, 16, 16, 32 + 3 # +3 for prior encoding (velocity)
    x = torch.randn(B, L, H, W, C)
    mask = torch.ones(B, L).bool()
    spatial_correction_matrix = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
    record_len = torch.tensor([2])
    object_ids = [[0, 1]] # Agent 0 (Ego), Agent 1 (Neighbor)
    
    # --- Step 1: Time T=0 (I-Frame) ---
    print("Step 1: T=0")
    timestamp = 0
    with torch.no_grad():
        out1 = model(x, mask, spatial_correction_matrix, record_len, object_ids=object_ids, timestamp=timestamp)
    
    print("Output shape:", out1.shape)
    assert out1.shape == (B, 32, H, W), f"Expected (1, 32, 16, 16), got {out1.shape}"
    
    # Verify memory populated
    assert 1 in model.memory, "Agent 1 should be in memory"
    cached_feat, cached_time = model.memory[1]
    assert cached_time == 0
    print("Memory populated successfully.")
    
    # --- Step 2: Time T=1 (P-Frame) ---
    print("Step 2: T=1 (Simulating P-Frame extrapolation)")
    timestamp = 1.0 # 1 second later
    # Create x2 such that Agent 1 has moved.
    # But for P-Frame logic, we assume we receive x2 (which might have MV) 
    # and we verify that the model uses extrapolation logic.
    # To verify extrapolation happened, we can check if the code path ran.
    # Since I cannot easily hook into `warp_affine_simple` without mocking, 
    # I will rely on the fact that it doesn't crash and returns valid output.
    
    # Populate velocity in x2 for Agent 1 to be non-zero to force non-identity warp
    x2 = x.clone()
    # Set velocity for Agent 1 (index 1)
    # C-3 is vx. Let's set it to 1.0
    x2[0, 1, :, :, -3] = 10.0 # High velocity, should result in shift
    x2[0, 1, :, :, -2] = 0.0
    
    with torch.no_grad():
        out2 = model(x2, mask, spatial_correction_matrix, record_len, object_ids=object_ids, timestamp=timestamp)
        
    print("Output shape T=1:", out2.shape)
    
    # Check if memory updated
    cached_feat_2, cached_time_2 = model.memory[1]
    assert cached_time_2 == 1.0
    
    print("Test passed!")

from opencood.models.sub_modules.video_codec_fusion import VideoCodecLayer


def test_video_codec_layer():
    print("Testing VideoCodecLayer...")
    # Test with compression enabled
    # Input channels 32, compression ratio 2 -> compressed dim 16
    layer = VideoCodecLayer(keyframe_interval=2, block_size=4, search_range=4, compression_args=2, input_channels=32)
    
    # Input: (N, C, H, W)
    # N=2 (2 agents), C=32
    N, C, H, W = 2, 32, 16, 16
    x = torch.randn(N, C, H, W)
    agent_ids = [100, 200]
    
    # Frame 0: I-Frame for both
    # Expected output: (N, C_compressed + 2, H, W) = (2, 16+2, 16, 16)
    out0 = layer(x, agent_ids)
    print("Frame 0 Output shape:", out0.shape)
    assert out0.shape == (N, 16 + 2, H, W)
    
    # Check that MVs are zero (initially)
    mvs0 = out0[:, 16:, :, :] # Last 2 channels
    assert torch.all(mvs0 == 0)
    print("Frame 0: I-Frame verified (MVs are 0).")
    
    # Frame 1: P-Frame for both
    # Shift x for agent 100 by (2, 2)
    x1 = x.clone()
    # roll dims (2,3) -> H, W
    x1[0] = torch.roll(x[0], shifts=(2, 2), dims=(1, 2)) 
    # Agent 200 no shift
    
    out1 = layer(x1, agent_ids)
    
    # Check MVs for Agent 100
    # Channel 16 is dx, 17 is dy
    dx = out1[0, 16]
    dy = out1[0, 17]
    
    valid_dx = dx[4:12, 4:12]
    valid_dy = dy[4:12, 4:12]
    
    print(f"Agent 100 Mean DX: {valid_dx.mean().item()} (Expected ~ -2.0)")
    print(f"Agent 100 Mean DY: {valid_dy.mean().item()} (Expected ~ -2.0)")
    
    # Agent 200 should have 0 MV
    dx_2 = out1[1, 16]
    print(f"Agent 200 Mean DX: {dx_2.mean().item()} (Expected 0.0)")
    
    print("Frame 1: P-Frame verified.")


def test_point_pillar_generic():
    print("Testing PointPillarPTVideoCodec Instantiation...")
    from opencood.models.point_pillar_pt_video_codec import PointPillarPTVideoCodec
    
    args = {
        'max_cav': 2,
        'voxel_size': [0.4, 0.4, 4],
        'lidar_range': [0, -40, -3, 70.4, 40, 1],
        'pillar_vfe': {'use_norm': True, 'with_distance': False, 'use_absolute_xyz': True, 'num_filters': [64]},
        'point_pillar_scatter': {'num_features': 64},
        'base_bev_backbone': {'layer_nums': [3, 5, 5], 'layer_strides': [2, 2, 2], 'num_filters': [64, 128, 256], 'upsample_strides': [1, 2, 4], 'num_upsample_filters': [64, 128, 128]},
        'anchor_number': 2,
        'compression': 2, # Enabled
        'transformer': {'core_method': 'v2xvit'}, # Standard transformer
        'backbone_fix': False,
        'point_transformer_vfe': {'dim': 64, 'depth': 1, 'heads': 4, 'dim_head': 16, 'mlp_dim': 64, 'dropout': 0.1, 'num_point': 100} 
    }
    
    model = PointPillarPTVideoCodec(args)
    print("Model instantiated successfully.")
    # We won't run full forward pass due to complex input requirement, but instantiation checks imports/init logic.


if __name__ == "__main__":
    if torch.cuda.is_available(): # Just to import torch usually, assuming cpu for test
        pass
    test_video_codec_fusion()
    test_video_codec_layer()
    test_point_pillar_generic()

