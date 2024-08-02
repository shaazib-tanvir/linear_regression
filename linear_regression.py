from torch import Tensor

class LinearRegression:
    def __init__(self, input_tensor: Tensor, output_tensor: Tensor):
        self._input_tensor = input_tensor
        self._output_tensor = output_tensor

    def train(self) -> None:
        input_tensor_transpose = self._input_tensor.transpose(0, 1)
        self._bias = input_tensor_transpose.matmul(self._input_tensor).inverse().matmul(input_tensor_transpose.matmul(self._output_tensor))

    def get(self, input_vector: Tensor):
        assert self._bias != None
        return input_vector.matmul(self._bias)
