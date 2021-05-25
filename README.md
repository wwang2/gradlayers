# gradlayers

Code to compute gradients/sensitivities of MLP inputs without calling .backward()

example:

```
from gradlayers.compose import TupleSequential
from gradlayers.layers import GradLayer, dtanh

gradMLP = TupleSequential( GradLayer(8, 8, activation=torch.nn.Tanh(), activation_grad=dtanh),
                           GradLayer(8, 4, activation=torch.nn.Tanh(), activation_grad=dtanh),
                           GradLayer(4, 1))

x = torch.randn(4, 8)
inputs = (x, None)

gradMLP(*inputs)
```
out: 

```
(tensor([[0.2526],
         [0.2152],
         [0.0856],
         [0.2645]], grad_fn=<AddBackward0>),
 tensor([[[-0.0759, -0.0207, -0.0266, -0.0511, -0.0237, -0.0878,  0.0052, 0.0436]],
 
         [[-0.0925, -0.0209, -0.0219, -0.0650, -0.0185, -0.0993,  0.0256, 0.0304]],
 
         [[-0.0725, -0.0230, -0.0166, -0.0617, -0.0305, -0.0835,  0.0396, 0.0225]],
 
         [[-0.0806, -0.0203, -0.0260, -0.0612, -0.0145, -0.0863,  0.0295, 0.0228]]], grad_fn=<TransposeBackward0>))
```

todo:

Add benchmarks to compare with out.backward()
Include more activation functions
More architectures like GNN
