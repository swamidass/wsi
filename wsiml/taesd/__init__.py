from typing import Hashable, Generator
import onnxruntime as rt
import wsiml.patch
import xarray as xr
import numpy as np
import pooch
import tqdm

model_path = pooch.retrieve(
    "https://github.com/swamidass/wsi/raw/master/data/taesd-encoder.onnx",
    known_hash="md5:1e64ceb4eb98bef77202c91f6584e50a",
)


class TAESD:
    dss = wsiml.patch.DownSamplerShape(padding=140, stride=8, odd=1)

    def __init__(self, model_path=model_path, **kwargs):
        self.model_path = model_path
        self.model = rt.InferenceSession(model_path, **kwargs)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def predict(self, x: xr.DataArray) -> xr.DataArray:

        p = self._predict(x.data)
        d = (x.dims[0], x.dims[1], "taesd")
        c = {
            dim: (dim, grid.downsample(self.dss).to_numpy(), x[dim].attrs)
            for dim, grid in x.wsi.grids.items()
        }

        return xr.DataArray(p, dims=d, coords=c)

    def _predict(self, x: np.ndarray[np.uint8]) -> np.ndarray[np.uint8]:
        return self.model.run([self.output_name], {self.input_name: x})[0]

    def tiled_predict(self, x: xr.DataArray, approx_patch_size: int = 1500 * 2) -> xr.DataArray:
        processed = []
        slicers = list(compute_patches(self.dss, x, approx_patch_size))
        for slicer in tqdm.tqdm(slicers):
            patch = x[slicer]
            pred = self.predict(patch)
            processed.append(pred)

        return xr.combine_by_coords(processed)  # type: ignore


def compute_patches(
    dss: wsiml.patch.DownSamplerShape,
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

    size = [g[d].size for d in dims]
    area = size[0] * size[1]
    patch_area = approx_patch_size**2

    # approximate number of patches of size approx_patch_size
    N = area / patch_area

    # aspect ratio
    r = size[0] / size[1]

    splits = [
        max(1, int(np.rint(N**0.5 * r**0.5))),  # type: ignore
        max(1, int(np.rint(N**0.5 / r**0.5))),  # type: ignore
    ]

    spacing = dss.stride

    padding = dss.padding * 2 + dss.odd

    y_stride = (size[0] - padding) // (splits[0] * spacing) * spacing
    x_stride = (size[1] - padding) // (splits[1] * spacing) * spacing

    y_side_mod = (size[0] - padding) % (splits[0] * spacing) // spacing
    x_side_mod = (size[1] - padding) % (splits[1] * spacing) // spacing

    y_size = y_stride + padding
    x_size = x_stride + padding

    for j in range(splits[0]):
        for i in range(splits[1]):
            x_start = i * x_stride + spacing * min(i, x_side_mod)
            y_start = j * y_stride + spacing * min(j, y_side_mod)

            x_end = x_start + x_size + (spacing if i < x_side_mod else 0)
            y_end = y_start + y_size + (spacing if j < y_side_mod else 0)

            slicer = {
                dims[0]: slice(y_start, y_end),
                dims[1]: slice(x_start, x_end),
            }
            yield slicer
