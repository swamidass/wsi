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
    """
    Simplified encoder from stable diffusion in ONNY format.
    """

    dss: wsiml.patch.DownSamplerShape = wsiml.patch.DownSamplerShape(padding=140, stride=8, odd=1)

    def __init__(self, model_path=model_path, **onnx_kwargs):
        self.model_path = model_path
        self.model = rt.InferenceSession(model_path, **onnx_kwargs)
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

    def tiled_predict(
        self, x: xr.DataArray, approx_patch_size: int = 1500 * 2
    ) -> xr.DataArray:
        processed = []
        slicers = list(self.dss.partition(x, approx_patch_size))
        for slicer in tqdm.tqdm(slicers):
            patch = x[slicer]
            pred = self.predict(patch)
            processed.append(pred)

        return xr.combine_by_coords(processed)  # type: ignore
