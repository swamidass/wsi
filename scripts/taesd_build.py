
from diffusers import AutoencoderTiny
import torch
import torch.nn as nn

taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd")

class TASEDCompressor(nn.Module):

    def __init__(self):
        super(TASEDCompressor, self).__init__()
        self.encoder = taesd.encoder  # type: ignore

    def forward(self, x):
        p = 18
        i = (x.shape[0] // 8) * 8
        j = (x.shape[1] // 8) * 8
        
        x = x[:i, :j, :]
        x = x / 255.0
        
        x = x.permute(2, 0, 1)
        
        x = self.encoder.forward(x)
        x = x.permute(1, 2, 0)
        x = x[p:-p, p:-p, :]
        x = taesd.scale_latents(x)
        x = (x * 255).type(torch.uint8)
        return x

m = TASEDCompressor()

export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(
    m,
    torch.zeros((1024,512,3), dtype=torch.uint8),
    export_options=export_options)
onnx_program.save("wsiml/taesd/encoder.onnx")
