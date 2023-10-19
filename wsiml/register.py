import tiffslide
import numpy as np
import os
import xarray as xr
import itk
from typing import Union, Optional

from typing import Optional
import pydantic



class RegisterWSI:
  downsample = 16

  def __init__(self):
    self.params = itkparams_to_dicts(get_default_params())

  def fit(self, fixed : xr.DataArray, moving : xr.DataArray):
    fixed = xarray_downsample(fixed, self.downsample) # type: ignore
    moving = xarray_downsample(moving, self.downsample) # type: ignore

    reg, trans = fit_transform(fixed, moving, dicts_to_itkparams(self.params))
    self.transform = itkparams_to_dicts(trans) # type: ignore
    return reg
  
  def transform_image(self, moving : xr.DataArray, downsample : Optional[int] = None):

    downsample = downsample if downsample else self.downsample

    moving = xarray_downsample(moving, downsample) # type: ignore

    t = dicts_to_itkparams(self.transform)
    return apply_transform(t, moving)



def infer_spacing(x: Union[xr.DataArray, xr.Dataset]) -> tuple[float, float]:
  """Infer the level of downsampling of this array, assuming that all coordinates are equally spaced."""
  return (float(x.x[1] - x.x[0]),  float(x.y[1] - x.y[0]))

def xarray_downsample(x : Union[xr.Dataset, xr.DataArray], downsample : int = 16) -> Union[xr.Dataset, xr.DataArray]:
  """Downsamples input x array/dataset to "downsample," taking account of how much it is already downsampleed. If no 
  downsampling needed, then returns x without any downsampling."""
  spacing = infer_spacing(x)

  if spacing == (downsample, downsample):
    return x
  
  else:
    xd = downsample / spacing[0]
    assert int(xd) == xd, f"input array sampling must be evenly divisible by downsample {downsample}"

    yd = downsample / spacing[0]
    assert int(yd) == yd, f"input array sampling must be evenly divisible by downsample {downsample}"

    xd = int(xd)
    yd = int(yd)

    x = x.chunk({"x": 256 * 16,"y": 256 * 16, 'rgb': 1})
    x = x.astype(np.float32).coarsen({"x": xd, "y": yd}, boundary="trim").mean() #type : ignore
    return x.compute()


def load_tiff_level(fname: str, level : int = 0) -> xr.Dataset:
  """Load specific level of a tiff slide into a xarray.Datset."""
  with tiffslide.TiffSlide(fname) as f:
    return _load_tiff_level(f, fname, level)


def _load_tiff_level(f: tiffslide.TiffSlide, fname, level : int = 0) -> xr.Dataset:
  """Eagerly load a particular level of a tiff slide. Add coordinates, attributes, and set encodings
  to reasonable defaults."""
  shape =  f.level_dimensions[level]
  
  downsample = f.level_downsamples[level]
  attr = {k: v for k,v in f.properties.items() if (v != None and "tiffslide.level" not in k)}

  assert attr['tiffslide.series-axes'] == "YXS"

  offset = (downsample - 1) / 2 if downsample > 1 else 0
  stride = downsample

  if int(downsample) == downsample: downsample = int(downsample)

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





def get_default_params() -> itk.ParameterObject: #type: ignore
  parameter_object = itk.ParameterObject.New()  #type: ignore
  P = parameter_object.GetDefaultParameterMap('rigid')
  P["NumberOfResolutions"] = ["15"] # high of resolution is *critical* for WSI
  P["NumberOfSpatialSamples"] = ["5000"]
  #P["NewSamplesEveryIteration"] = ["false"]
  parameter_object.AddParameterMap(P)

  P = parameter_object.GetDefaultParameterMap('bspline')
  P["FinalGridSpacingInPhysicalUnits"] = [str(16 * 8)]
  parameter_object.AddParameterMap(P)
  return parameter_object


def itk_to_xarray(img : itk.Image) -> xr.DataArray: #type: ignore
  arr =  itk.GetArrayFromImage(img)

  coords = {}
  for origin, spacing, size, k in zip(img.GetOrigin(), img.GetSpacing(), arr.shape, ['y', 'x']):
    coords[k] = np.arange(size) * spacing + origin

  arr = xr.DataArray(arr, dims=('y', 'x'), coords=coords)
    
  return arr



def _registration_image(fixed : itk.Image, out : itk.Image) -> xr.DataArray: #type: ignore
    import numpy as np
    r = itk_to_xarray(fixed) 
    g = itk_to_xarray(out) 
    b = np.zeros_like(r)

    reg = np.array([r,g,b]).transpose([1,2,0])
    reg = xr.DataArray(reg, dims=('y', 'x', 'registration'), coords={"x": r.x, "y": r.y, "registration": ['fixed', 'moving', 'zeros', ]})
    reg = reg.clip(0,255)
    reg = reg.astype(np.uint8)
    return reg

def _xarray_to_intensities(arr, dtype=np.int16):


    
  arr = arr.astype(float)
  arr = arr.sum(dim='rgb')
  arr = (255 * 3 - arr) 
  arr = arr.astype(dtype=dtype)

  spacing = infer_spacing(arr)
  origin = [(c-1) / 2 for c in spacing]

  img =  itk.GetImageFromArray(arr)

  img.SetSpacing(tuple(spacing))
  img.SetOrigin(tuple(origin))
  return img


def fit_transform(fixed : xr.DataArray, moving : xr.DataArray, PARAMS : Optional[itk.ParameterObject]= None)  -> tuple[xr.Dataset, itk.ParameterObject]: #type: ignore
  fixed = _xarray_to_intensities(fixed)
  moving = _xarray_to_intensities(moving)
  #moving =  out

  PARAMS = PARAMS if PARAMS else get_default_params()

  elastix_object = itk.ElastixRegistrationMethod.New() #type: ignore
  elastix_object.SetFixedImage(fixed)
  elastix_object.SetMovingImage(moving)
  elastix_object.SetParameterObject(PARAMS)
  elastix_object.SetLogToConsole(False)

  elastix_object.UpdateLargestPossibleRegion()

  out = elastix_object.GetOutput()
  result_transform_parameters : itk.ParameterObject = elastix_object.GetTransformParameterObject() #type: ignore

  reg = _registration_image(fixed, out)
  return reg, result_transform_parameters


def itkparams_to_dicts(params : itk.ParameterObject) -> list[dict[str, tuple]]: #type: ignore
  out = []
  for i in range(params.GetNumberOfParameterMaps()):
    p = params.GetParameterMap(i)
    out.append(dict(p))

  return out

def dicts_to_itkparams(dicts:  list[dict[str, tuple]]) -> itk.ParameterObject: #type: ignore
  out = itk.ParameterObject.New() #type: ignore
  
  for d in dicts:
    p = itk.elxParameterObjectPython.mapstringvectorstring() #type: ignore
    for k,v in d.items(): p[k] = v
    out.AddParameterMap(p)
  
  return out


def xarray_to_itk(arr : xr.DataArray, dtype=np.int16) -> itk.Image: #type: ignore
  spacing = []
  for i  in range(2):
    c = arr.dims[i]
    c = arr[c]
    spacing.append(float(c[1]) - float(c[0]))

  origin = [(c-1) / 2 for c in spacing]

  img =  itk.GetImageFromArray(np.asarray(arr, dtype=dtype))

  img.SetSpacing(tuple(spacing))
  img.SetOrigin(tuple(origin))
  return img


def apply_transform(tx, moving_image : xr.DataArray, spacing : tuple =None, origin : tuple =None, size : tuple=None, default_pixel : int=255):
    d = itkparams_to_dicts(tx)

    x = moving_image.x
    y = moving_image.y

    fixed_origin = np.array([float(x) for x in d[-1]["Origin"]])
    fixed_spacing = np.array([float(x) for x in d[-1]["Spacing"]]) 
    fixed_size = np.array([float(x) for x in d[-1]["Size"]]) 

    moving_spacing = np.array([x[1] - x[0], y[1] - y[0]])
    
    spacing = np.array(spacing) if spacing is not None else moving_spacing
    
    origin = np.array(origin) if origin is not None else fixed_origin - (fixed_spacing - 1) / 2
    origin = origin + (spacing - 1) / 2

    size = np.array(size) if size is not None else fixed_size * fixed_spacing
    size = (size/ spacing).astype(np.int32)

    d[-1]["Origin"]  = (str(origin[0]),str(origin[1]))
    d[-1]["Spacing"]  = (str(spacing[0]),str(spacing[1]))
    d[-1]["Size"]  = (str(size[0]),str(size[1]))

    
    p = dicts_to_itkparams(d)

    T = itk.TransformixFilter.New()
    T.SetTransformParameterObject(p)

    channel_dim = moving_image.dims[-1]
    channel_n = moving_image.shape[-1]

    out = []
      
    for i in range(channel_n):
        x = xarray_to_itk(default_pixel - moving_image[...,i])
        T.SetInput(x)
        T.SetLogToConsole(False)
        x = itk_to_xarray(T.GetOutput())
        x = (default_pixel - x).clip(0, 255).astype(np.uint8)
        out.append(x)
        del x
        
    channel_dim = moving_image.dims[-1]
    out = xr.concat(out, dim=channel_dim).transpose('y', 'x', channel_dim)
    out[channel_dim] = moving_image[channel_dim]

    return out


def transform_match_region(tfm, moving : xr.DataArray, region : xr.DataArray) -> xr.DataArray: 

  spacing = region.x[1] - region.x[0], region.y[1] - region.y[0]
  origin = region.x[0] - (spacing[0] -1)/ 2, region.y[0] - (spacing[1] -1)/ 2
  size = len(region.x), len(region.y) 

  return apply_transform(tfm, moving, origin=origin, spacing=spacing, size=size)



### save trasnform...
# result_transform_parameters.WriteParameterFiles(result_transform_parameters, ["temp1.txt", "temp2.txt"])
# P = itk.ParameterObject.New()
# P.ReadParameterFiles(["temp1.txt", "temp2.txt"])
# print(P)