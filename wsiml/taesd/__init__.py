import onnxruntime as rt

import pooch


model_path = pooch.retrieve(
    "https://github.com/swamidass/wsi/raw/master/data/taesd-encoder.onnx",
    known_hash="md5:1e64ceb4eb98bef77202c91f6584e50a",
)


class TAESD:
    def __init__(self, model_path=model_path):
        self.model_path = model_path
        self.model = rt.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def predict(self, x):
        return self.model.run([self.output_name], {self.input_name: x})[0]
    
