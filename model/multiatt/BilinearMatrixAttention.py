import paddle

class BilinearMatrixAttention(paddle.nn.Layer):
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False,
                 label_dim: int = 1,):
        super().__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1
        if label_dim == 1:
            self._weight_matrix = paddle.create_parameter(shape=[matrix_1_dim, matrix_2_dim], dtype='float32',is_bias=True,default_initializer=paddle.nn.initializer.XavierUniform())
        else:
            self._weight_matrix = paddle.create_parameter(shape=[label_dim,matrix_1_dim, matrix_2_dim], dtype='float32',is_bias=True,default_initializer=paddle.nn.initializer.XavierUniform())
        self._bias = paddle.create_parameter(shape=[1],dtype='float32',default_initializer=paddle.nn.initializer.Constant(0.0))
    def forward(self,matrix_1,matrix_2):
        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = paddle.matmul(matrix_1.unsqueeze(1), weight)
        final = paddle.matmul(intermediate, matrix_2.unsqueeze(1).transpose([0,1,3,2]))
        return final.squeeze(1)+self._bias


# a = [
#     [[1,2,5],[3,4,5]],
#     [[3,3,3],[8,7,5]],
#     [[3,3,3],[8,7,5]]
# ]
#
# b = [
#     [[1,2,2],[3,4,3]],
#     [[1,3,4],[3,4,5]],
#     [[3,3,3],[8,7,5]]
# ]
# a = paddle.to_tensor(a,dtype='float32')
# b = paddle.to_tensor(b,dtype='float32')
# print('a',a.shape)
# f = BilinearMatrixAttention(
#     matrix_1_dim=3,
#     matrix_2_dim=3,
# )
#
# print(f(a,b))




