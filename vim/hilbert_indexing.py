import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers.helpers import to_2tuple
from hilbertcurve.hilbertcurve import HilbertCurve

class Patchify(nn.Module):
    """ 2D Image to Patch Embedding WITH Hilbert Indexing
    """
    def __init__(self, n_bits=4, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Linear(in_chans * patch_size[0] * patch_size[1], embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.n_bits = n_bits
        self.hilbert = HilbertCurve(p=n_bits, n=2)
        self.num_patches = (2 ** n_bits) ** 2
        self.index_map = torch.tensor(self.hilbert.points_from_distances(list(range(self.num_patches)))).t()
        self.map_index = torch.zeros(2 ** n_bits, 2 ** n_bits, dtype=int)
        for t in range(self.num_patches):
            self.map_index[self.index_map[0][t], self.index_map[1][t]] = t
        
    def flatten_patches(self, patch_map):
        B, C, H, W = patch_map.shape
        assert H == W == 2 ** self.n_bits
        flattened = patch_map[:, :, self.index_map[0], self.index_map[1]]
        flattened = flattened.transpose(1, 2)
        assert flattened.shape == (B, H * W, C)
        return flattened.cuda()
    
    def unflatten_patches(self, patch_serial):
        B, N, C = patch_serial.shape
        L = int(N ** 0.5)
        assert N == self.num_patches
        patch_serial = patch_serial.transpose(1, 2)
        unflattened = patch_serial[:, :, self.map_index]
        assert unflattened.shape == (B, C, L, L)
        return unflattened.cuda()
    
    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = self.patch_size
        assert H == self.img_size[0] and W == self.img_size[1]
        x = rearrange(x, 'b c (h p) (w q) -> b (c p q) h w', p=ph, q=pw)
        x = self.flatten_patches(x)
        x = self.proj(x)
        x = self.norm(x)
        return x
