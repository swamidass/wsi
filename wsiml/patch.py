from __future__ import annotations

from typing import Any, Generator
import numpy as np
import xarray as xr
from tiffslide_xarray.grid import RegularGrid, DownSamplerShape
from itertools import product

KerasModel = Any

def _partition(self, dss: DownSamplerShape,  approx_size: int)   -> Generator[slice, Any, None]:
    size = self.size
    assert size is not None

    splits = max(1, int(np.rint(size / approx_size))) #type: ignore
    spacing = dss.stride
    padding = dss.padding * 2 + dss.odd

    stride = (size - padding) // (splits * spacing) * spacing
    strid_mod = (size - padding) % (splits * spacing) // spacing

    patch_size = stride + padding

    for i in range(splits):
        patch_size = stride + padding + (spacing if i < strid_mod else 0)
        start = i * stride + spacing * min(i, strid_mod)
        end = start + patch_size
        yield slice(start, end)

# monkey patch Regular Grid with new method
RegularGrid.partition = _partition

def _partition_image(
    self: DownSamplerShape,
    D: xr.DataArray,
    approx_patch_size: int = 1500 * 2,
):
    """
    Compute patches that partition input image and can be downsampled and combined into a single image identical
    to running downsampler on the full image.

    Parameters
    ----------
    dss : wsiml.patch.DownSamplerShape
        The downsampler shape to define the overlap between tiles.
    D : xr.DataArray
        The data array to partition.
    approx_patch_size : int, optional
        The target patch size to aim for, by default 3000

    Yields
    ------
    Generator[dict[Hashable, slice], None, None]
        slicers to index the data array into patches that can be downsampled andcombined by coordinates.

    """
    g = D.wsi.grids
    dims = D.dims[:2]

    for i, j in product(
        g[dims[0]].partition(self, approx_patch_size),
        g[dims[1]].partition(self, approx_patch_size),
    ):
        slicer = {
            dims[0]: i,
            dims[1]: j,
        }
        yield slicer


# monkey patch with new method
DownSamplerShape.partition = _partition_image
