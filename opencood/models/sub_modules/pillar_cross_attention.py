import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import gc
from torch.utils.checkpoint import checkpoint

class PillarCrossAttention(nn.Module):
    def __init__(self, model_cfg, num_pillar_features):
        '''Pillar Cross Attention Module
            It is a module that takes the input features of the pillars and applies the cross attention mechanism,
            relative to the input features of the pillars themselves. It applies a kernel like operation to the input
            features of the pillars and computes the cross attention relative to the neighboring pillars
        Args:
            model_cfg (dict): Dictionary containing the configuration of the model
            num_pillar_features (int): Number of input features of the pillars
            num_output_CA_features (int): Number of additional output features of the Pillar Cross Attention Module
        '''
        
        super().__init__()
        self.model_cfg = model_cfg
        self.num_pillar_features = num_pillar_features

        self.num_output_CA_features = self.model_cfg['num_output_CA_features']
        self.kernel_size = self.model_cfg['kernel_size']

        # Linear layers for query, key and value
        self.linear_query = nn.Linear(self.num_pillar_features, self.num_output_CA_features)
        self.linear_key = nn.Linear(self.num_pillar_features, self.num_output_CA_features)
        self.linear_value = nn.Linear(self.num_pillar_features, self.num_output_CA_features)

        # Learnable positional encoding
        self.positional_encoding = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, self.num_pillar_features)
        )
        
        self.is_mask = False

    def positional_encoder(self, device):
        '''This function generates the positional encoding for the cross attention mechanism
        It does so by taking the coordinates of a tensor similar to the kernel and generates position encoding that 
        is added to the input features of the pillars
        Args:
            kernel_size (int): Kernel size of the cross attention mechanism
            num_pillar_features (int): Number of input features of the pillars
        Returns:
            torch.Tensor: Sinusoidal positional encoding
        '''
        
        coordinates = torch.arange(self.kernel_size).float().repeat(self.kernel_size, 1).to(device)
        coordinates = torch.cat([coordinates.t().unsqueeze(-1), coordinates.unsqueeze(-1)], dim=-1)
        coordinates = coordinates.reshape(-1, 2).requires_grad_(False)
        position_encoder = self.positional_encoding(coordinates)

        return position_encoder
    
    def create_mask(self, H, W, kernel_size):
        '''Create Mask
            It creates a mask for the cross attention mechanism
        Args:
            H (int): Height of the mask
            W (int): Width of the mask
            kernel_size (int): Kernel size of the cross attention mechanism
        Returns:
            torch.Tensor: Mask
        '''
        
        mask = torch.ones(H, W).unsqueeze(0).unsqueeze(1)
        mask = F.unfold(mask, (kernel_size, kernel_size), stride=1)
        mask = F.fold(mask, (H, W), (kernel_size, kernel_size), stride=1)
        mask = mask.requires_grad_(False)
        self.is_mask = True

        return mask


    def forward(self, batch_dict):
        # # 1. Retrieve the input pseudo_image and related information
        # pseudo_image = batch_dict['spatial_features']  # [B, C, H, W]
        # B, C, H, W = pseudo_image.shape
        # device = pseudo_image.device

        # # 2. Initialize a tensor for storing cross attention results
        # cross_attention_result = torch.zeros(B, self.num_output_CA_features, H, W, device=device)

        # # 3. If the mask has not been generated, create the normalization mask and save it
        # if not self.is_mask:
        #     self.cross_attention_map = self.create_mask(H, W, self.kernel_size).to(device)
        # # The mask will be used later for normalization

        # # 4. Compute the quotient and remainder for height and width based on kernel_size
        # length_quotient = H // self.kernel_size
        # length_remainder = H % self.kernel_size
        # width_quotient = W // self.kernel_size
        # width_remainder = W % self.kernel_size

        # # Flags to adjust the quotient only once when the remainder is exceeded
        # is_reduce_length = False
        # is_reduce_width = False

        # # 5. Define an inner function to compute cross attention for a local region
        # def cross_attention_compute(kern, pos_enc):
        #     # kern: [B, num_patches, kernel_area, num_features]
        #     # pos_enc: [kernel_area, num_pillar_features] (will be broadcast to kern)
        #     kern = kern + pos_enc  # Add positional encoding
        #     query = self.linear_query(kern)
        #     key = self.linear_key(kern)
        #     value = self.linear_value(kern)
        #     # Compute attention: dot product, softmax, then multiplication by value
        #     attn = torch.matmul(query, key.transpose(2, 3))
        #     attn = F.softmax(attn, dim=-1)
        #     attn = torch.matmul(attn, value)
        #     return attn

        # # 6. Initialize a tensor to accumulate the attention results (same size as cross_attention_result)
        # cross_attention_accum = torch.zeros_like(cross_attention_result, device=device)

        # # 7. Nested loops: iterate over each offset position within the kernel
        # for i in range(self.kernel_size):
        #     for j in range(self.kernel_size):
        #         # Adjust the patch count when exceeding the remainder (reduce quotient only once)
        #         if i > length_remainder and not is_reduce_length:
        #             length_quotient -= 1
        #             is_reduce_length = True
        #         if j > width_remainder and not is_reduce_width:
        #             width_quotient -= 1
        #             is_reduce_width = True

        #         # 7.1 Extract the corresponding local region from pseudo_image based on the current offset (i, j)
        #         # The region spans [i : i + kernel_size * length_quotient] in height and [j : j + kernel_size * width_quotient] in width
        #         patch = pseudo_image[:, :,
        #                             i : i + self.kernel_size * length_quotient,
        #                             j : j + self.kernel_size * width_quotient]
        #         # 7.2 Rearrange the patch into shape [B, num_patches, kernel_area, num_features]
        #         # where kernel_area = self.kernel_size * self.kernel_size,
        #         # and num_patches = (length_quotient * width_quotient)
        #         kernel = einops.rearrange(
        #             patch,
        #             'b c (l k1) (w k2) -> b (l w) (k1 k2) c',
        #             k1=self.kernel_size,
        #             k2=self.kernel_size
        #         ).to(device)

        #         # 7.3 Get the fixed positional encoding for the local region, shape: [kernel_area, num_pillar_features]
        #         pos_enc = self.positional_encoder(device).to(device)

        #         # 7.4 Compute local attention using checkpoint (the inner function must be side-effect free)
        #         if torch.is_grad_enabled():
        #             attn_patch = checkpoint(cross_attention_compute, kernel, pos_enc)
        #         else:
        #             attn_patch = cross_attention_compute(kernel, pos_enc)

        #         # 7.5 Rearrange the attention result back to the original spatial layout:
        #         # Convert from [B, num_patches, kernel_area, num_features] back to [B, num_features, l*k1, w*k2]
        #         attn_patch = einops.rearrange(
        #             attn_patch,
        #             'b (l w) (k1 k2) c -> b c (l k1) (w k2)',
        #             l=length_quotient,
        #             k1=self.kernel_size
        #         ).to(device)

        #         # 7.6 Accumulate the attention result in the corresponding region
        #         cross_attention_accum[:, :,
        #                             i : i + self.kernel_size * length_quotient,
        #                             j : j + self.kernel_size * width_quotient] += attn_patch

        # # 8. Normalize the accumulated attention by dividing by the pre-computed mask
        # cross_attention_accum /= self.cross_attention_map

        # # 9. Concatenate the original pseudo_image with the computed cross attention features along the channel dimension
        # batch_dict['spatial_features'] = torch.cat([pseudo_image, cross_attention_accum], dim=1)

        # return batch_dict

        '''Forward pass of the Pillar Cross Attention Module
        '''
        pseudo_image = batch_dict['spatial_features']
        B, C, H, W = pseudo_image.shape
        cross_attention = torch.zeros(B, self.num_output_CA_features, H, W).to(pseudo_image.device)
        self.cross_attention_map = self.create_mask(H, W, self.kernel_size).to(pseudo_image.device) if not self.is_mask else self.cross_attention_map

        length_quotient = H // self.kernel_size
        length_remainder = H % self.kernel_size
        is_reduce_length = False
        width_quotient = W // self.kernel_size
        width_remainder = W % self.kernel_size
        is_reduce_width = False

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                
                # Extract the kernel
                if i > length_remainder and not is_reduce_length:
                    length_quotient -= 1
                    is_reduce_length = True
                
                if j > width_remainder and not is_reduce_width:
                    width_quotient -= 1
                    is_reduce_width = True
                
                kernel = pseudo_image[:, :, i:i+self.kernel_size*length_quotient, j:j+self.kernel_size*width_quotient].to(pseudo_image.device)
                kernel = einops.rearrange(kernel, 'b c (l k1) (w k2) -> b (l w) (k1 k2) c', k1 = self.kernel_size, k2 = self.kernel_size)

                # Compute the positional encoding
                positional_encoder = self.positional_encoder(pseudo_image.device)
                kernel += positional_encoder.to(pseudo_image.device)

                # Compute the query, key and value
                query = self.linear_query(kernel)
                key = self.linear_key(kernel)
                value = self.linear_value(kernel)


                # Compute the attention
                attention = torch.matmul(query, key.transpose(2, 3))
                attention = F.softmax(attention, dim=-1)

                # Compute the output
                attention = torch.matmul(attention, value).to(pseudo_image.device)
                attention = einops.rearrange(attention, 'b (l w) (k1 k2) c -> b c (l k1) (w k2)', l=length_quotient, k1 = self.kernel_size)

                # Update the pseudo image
                cross_attention[:, :, i:i+self.kernel_size*length_quotient, j:j+self.kernel_size*width_quotient] += attention
        
        # Normalize the pseudo image
        cross_attention /= self.cross_attention_map

        # Concatenate the cross attention with the pseudo image
        batch_dict['spatial_features'] = torch.cat([pseudo_image, cross_attention], dim=1)

        return batch_dict

        


    

    # def forward(self, batch_dict):
    #     '''Forward pass of the Pillar Cross Attention Module
    #     '''
    #     torch.cuda.empty_cache()
    #     kernel = batch_dict['spatial_features']
    #     B, C, H, W = kernel.shape

    #     kernel = F.unfold(kernel, (self.kernel_size, self.kernel_size), stride=1).to(kernel.device)
    #     kernel = kernel.permute(0, 2, 1).reshape(B, -1, self.kernel_size * self.kernel_size)
    #     kernel = einops.rearrange(kernel,'b (no_kernels d) k -> b no_kernels k d', d=self.num_pillar_features)
    #     kernel += self.positional_encoding.to(kernel.device)

    #     query = self.linear_query(kernel)
    #     key = self.linear_key(kernel)
    #     value = self.linear_value(kernel)

    #     kernel = torch.matmul(query, key.transpose(2, 3))
    #     kernel = F.softmax(kernel, dim=-1)

    #     del query
    #     del key
    #     gc.collect()

    #     kernel = torch.matmul(kernel, value)

    #     del value
    #     gc.collect()

    #     kernel = kernel.view(B, -1, self.num_output_CA_features, self.kernel_size * self.kernel_size)
    #     kernel = einops.rearrange(kernel, 'b no_kernels d k -> b no_kernels (d k)', d=self.num_output_CA_features).permute(0, 2, 1)
    #     kernel = F.fold(kernel, (H, W), (self.kernel_size, self.kernel_size), stride=1) 

    #     if not self.is_mask:
    #         self.cross_attention_mask = self.create_mask(H, W, self.kernel_size).to(kernel.device)
    #         self.is_mask = True
        
    #     kernel /= self.cross_attention_mask

        

    #     # Concatenate the cross attention with the pseudo image
    #     batch_dict['spatial_features'] = torch.cat([batch_dict['spatial_features'], kernel], dim=1)

    #     return batch_dict