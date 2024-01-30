import torch
import torch.nn as nn
from timm.models.layers.helpers import to_2tuple
from hilbertcurve.hilbertcurve import HilbertCurve

class Patchify(nn.Module):
    """ 2D Image to Patch Embedding WITH Hilbert Indexing
    """
    def __init__(self, n_bits=4, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.n_bits = n_bits
        self.hilbert = HilbertCurve(p=n_bits, n=2)
        self.n_patches = (2 ** n_bits) ** 2
        self.index_map = torch.tensor(self.hilbert.points_from_distances(torch.arange(self.n_patches)))
        
    def flatten_patches(self, patch_map):
        B, C, H, W = patch_map.shape
        assert H == W == 2 ** self.n_bits
        patch_serial = torch.zeros(B, H * W, C)
        for t in range(self.n_patches):
            patch_serial[:, t, :] = patch_map[:, :, self.index_map[t][0], self.index_map[t][1]]
        return patch_serial.cuda()
    
    def unflatten_patches(self, patch_serial):
        B, N, C = patch_serial.shape
        L = int(N ** 0.5)
        assert N == self.n_patches
        patch_map = torch.zeros(B, C, L, L)
        for t in range(self.n_patches):
            patch_map[:, :, self.index_map[t][0], self.index_map[t][1]] = patch_serial[:, t, :]
        return patch_map
    
    def forward(self, x):
        H, W = x.shape[-2:]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten: # BCHW -> BNC
            x = self.flatten_patches(x)
            # x = x.flatten(-2, -1).transpose(1, 2)
        x = self.norm(x)
        return x
