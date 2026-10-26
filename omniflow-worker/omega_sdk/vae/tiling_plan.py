"""SYS-AETHERIS-VISION-VAE-017: conservative, memory-bounded tile scheduling.

This plans neural inference, but does not execute the model. Coverage is
verified via the disjoint core partition, in O(number_of_tiles) memory.
The VRAM estimate is a heuristic multiplier, not a profiled memory bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isqrt

class TilingError(ValueError):
    pass

@dataclass(frozen=True)
class Tile:
    tile_id: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int
    padded_y0: int
    padded_y1: int
    padded_x0: int
    padded_x1: int

@dataclass(frozen=True)
class TilePlan:
    image_size: tuple[int,int]
    tile_core: int
    halo: int
    stride: int
    latent_shape: tuple[int,int,int,int,int]
    latent_bytes: int
    estimated_peak_vram_bytes: int
    tiles: tuple[Tile,...]

class TilePlanner:
    MAX_PIXELS = 500_000_000
    MAX_TILES = 200_000
    ACTIVATION_MULTIPLIER = 24   # empirical placeholder until real GPU profiling

    @staticmethod
    def build(height:int,width:int,*,tile_core:int=512,halo:int=64,
              stride:int=8,channels:int=16,bytes_per_latent:int=2,
              base_channels:int=64,vram_budget_bytes:int=8<<30)->TilePlan:
        nums=(height,width,tile_core,halo,stride,channels,bytes_per_latent,
              base_channels,vram_budget_bytes)
        if any(type(v) is not int for v in nums):
            raise TilingError('integer dimensions and budgets required')
        if (not (0<height and 0<width and height*width<=TilePlanner.MAX_PIXELS)
                or stride<1 or channels<1 or bytes_per_latent not in (2,4)
                or base_channels<1 or tile_core<stride or halo<0
                or tile_core%stride or halo%stride or vram_budget_bytes<=0):
            raise TilingError('invalid image/tile alignment or resource budget')
        side=tile_core+2*halo
        peak=side*side*base_channels*4*TilePlanner.ACTIVATION_MULTIPLIER
        if peak>vram_budget_bytes:
            max_side=isqrt(vram_budget_bytes//(base_channels*4*TilePlanner.ACTIVATION_MULTIPLIER))
            max_core=((max_side-2*halo)//stride)*stride
            if max_core<stride: suggested='none'
            else:suggested=str(max_core)
            raise TilingError(f'RESOURCE_EXHAUSTED: tile peak={peak} > budget={vram_budget_bytes}; suggested_core={suggested}')
        nh=(height+tile_core-1)//tile_core
        nw=(width+tile_core-1)//tile_core
        if nh*nw>TilePlanner.MAX_TILES:
            raise TilingError('tile count exceeds bounded control-plane budget')
        latent_shape=(1,channels,1,(height+stride-1)//stride,(width+stride-1)//stride)
        latent_bytes=channels*latent_shape[3]*latent_shape[4]*bytes_per_latent
        tiles=[]
        for yi in range(nh):
            y0=yi*tile_core;y1=min(y0+tile_core,height)
            for xi in range(nw):
                x0=xi*tile_core;x1=min(x0+tile_core,width)
                tiles.append(Tile(len(tiles),y0,y1,x0,x1,
                                  max(0,y0-halo),min(height,y1+halo),
                                  max(0,x0-halo),min(width,x1+halo)))
        plan=TilePlan((height,width),tile_core,halo,stride,latent_shape,
                      latent_bytes,peak,tuple(tiles))
        TilePlanner.verify_coverage(plan)
        return plan

    @staticmethod
    def verify_coverage(plan:TilePlan)->bool:
        """Prove exactly-once core coverage, without dense 400 MP mask allocation.

        The padded/halo region may overlap and still covers the same pixels
        because each core is entirely contained in its padded region.
        """
        if not isinstance(plan,TilePlan):
            raise TilingError('tile plan required')
        h,w=plan.image_size
        if min(h,w)<=0 or plan.tile_core<=0:
            raise TilingError('invalid plan geometry')
        nh=(h+plan.tile_core-1)//plan.tile_core
        nw=(w+plan.tile_core-1)//plan.tile_core
        if len(plan.tiles)!=nh*nw:
            raise TilingError('missing/duplicated tiles')
        for yi in range(nh):
            y0=yi*plan.tile_core;y1=min(y0+plan.tile_core,h)
            for xi in range(nw):
                x0=xi*plan.tile_core;x1=min(x0+plan.tile_core,w)
                index=yi*nw+xi
                tile=plan.tiles[index]
                if (tile.tile_id!=index or (tile.core_y0,tile.core_y1,tile.core_x0,tile.core_x1)!=(y0,y1,x0,x1)
                    or not (0<=tile.padded_y0<=y0<y1<=tile.padded_y1<=h)
                    or not (0<=tile.padded_x0<=x0<x1<=tile.padded_x1<=w)):
                    raise TilingError('gap/overlap or malformed tile/padding coordinates')
        return True
