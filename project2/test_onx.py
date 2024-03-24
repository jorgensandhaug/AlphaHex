import onnxruntime as ort
import torch.nn as nn
import torch

test_input = torch.randn(1, 256, 256, 256)

b = nn.GroupNorm(32, 256)
test_output = b(test_input)
torch.onnx.export(b, test_input, "group_norm.onnx", verbose=False, opset_version=17)

sess = ort.InferenceSession("group_norm.onnx", providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {sess.get_inputs()[0].name: test_input.detach().numpy()})
torch.testing.assert_close(torch.from_numpy(onnx_out[0]), test_output)