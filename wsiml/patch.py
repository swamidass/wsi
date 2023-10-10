from typing import NamedTuple

class ModelShape(NamedTuple):
  pad : int
  stride : int

def model_shape(size_function, start : int= 1024) -> ModelShape:
  breaks = []
  i = start
  last = size_function(i)
  while len(breaks) < 2:
    this = size_function(i)
    if this != last:
      breaks.append((i, this))
    last = this
    i += 1

  stride = breaks[1][0] - breaks[0][0]
  min_size = breaks[1][0]  - (breaks[1][1] - 1)  * stride
  pad = min_size // 2
  return ModelShape(stride=stride, pad=pad)

   

def _keras_config_change_padding(config, padding="valid"):
  """
  Convert Keras model config into an equivalent config that uses "valid."
  """
  
  if isinstance(config, list):
    return [_keras_config_change_padding(x) for x in config]

  if isinstance(config, tuple):
    return tuple([_keras_config_change_padding(x) for x in config])

  if isinstance(config, dict):
    if "padding" in config:
      config["padding"] = padding
  
    for k in list(config):
      config[k] = _keras_config_change_padding(config[k])
    
    return config
  
  return config
      
