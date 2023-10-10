import tiffslide
import numpy as np
import os
import xarray as xr
from datatree import DataTree
from typing import Optional

from xarray.backends import BackendEntrypoint


def load_tiff_level(fname: str, level : int = 0) -> xr.Dataset:
  """Load specific level of a tiff slide into a xarray.Datset."""
  with tiffslide.TiffSlide(fname) as f:
     return _load_tiff_level(f, fname, level)

def load_tiff_tree(fname: str) -> DataTree:
   """Load all levels of a tiff slide into a datatree.DataTree."""
   with tiffslide.TiffSlide(fname) as f:
      tree = {}
      
      for level in range(len(f.level_downsamples)):

        x = _load_tiff_level(f, fname, level)   

        tree["/" if level == 0 else f"level{level}"] = x

      tree = DataTree.from_dict(tree)
      return tree
         

def _load_tiff_level(f: tiffslide.TiffSlide, fname, level : int = 0) -> xr.Dataset:
  """Eagerly load a particular level of a tiff slide. Add coordinates, attributes, and set encodings
  to reasonable defaults."""
  shape =  f.level_dimensions[level]
   
  downsample = f.level_downsamples[level]
  attr = {k: v for k,v in f.properties.items() if (v != None and "tiffslide.level" not in k)}

  assert downsample== int(downsample)
  downsample = int(downsample)

  assert attr['tiffslide.series-axes'] == "YXS"

  offset = (downsample - 1) / 2 if downsample > 1 else 0
  stride = int(downsample)

  coords = {
      'y': ('y', np.arange(shape[1]) * stride + offset),
      'x': ('x', np.arange(shape[0]) * stride + offset),
      'rgb': ("rgb", ['r', 'g', 'b']),
    }

  x = f.read_region((0,0), level, shape, as_array=True)
  x = xr.DataArray(
    x,
    dims=('y', 'x', 'rgb'), 
    coords=coords)

  x.attrs["level"] = level
  x.encoding["chunksizes"] = (256, 256, 1)
  x.encoding["compression"] = "lzf"

  x = x.to_dataset(name="image")
  x.attrs.update(attr)

  if type(fname) == str:
    x.attrs["source_file"] = fname
    x["image"].attrs["source_file"] = fname

  for c in 'xy':
    x[c].encoding["scale_factor"] = stride
    x[c].encoding["add_offset"] = offset
    x[c].encoding["compression"] = "lzf"
    x[c].attrs["unit"] = "pixel"
  

  return x



class TiffBackendEntrypoint(BackendEntrypoint):
    """Add entry point for xarray so that xarray.open_dataset
    can eagerly load tiff and svs files using tiffslide. The keyword argument 
    "level" specifies which level of the image is read. Lazy loading
    not currently possible."""
  
    open_dataset_parameters : list  = ["level"]
    description : str  = "Load any image file compatible with Tiffslide."
    EXTENSIONS : set[str] = {".svs", ".tiff"}

    def open_dataset(
        self,
        filename_or_obj,
        level : Optional[int] = 0,
        *,
        drop_variables = None,
    ):
        assert type(level) == int
        return load_tiff_level(filename_or_obj)
        

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj) #type: ignore
        except TypeError:
            return False
        return ext in self.EXTENSIONS
