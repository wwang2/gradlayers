from torch.nn import init
from torch.nn.parameter import Parameter
    
class GradLayer(torch.nn.Linear):
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        activation_grad=None,
    ):
        
        super().__init__(in_features, out_features, bias)
        
        self.activation = activation
        self.activation_grad = activation_grad
        
    
    def forward(self, x, x_grad=None):

        z = x.matmul(self.weight.t()) + self.bias
        
        if x_grad == None:   
            x_grad_out = self.activation_grad(z).unsqueeze(-1) * self.weight.unsqueeze(0)
        else:
            x_grad_out = self.weight.matmul(x_grad) 
            if self.activation_grad:
                x_grad_out *= self.activation_grad(z).unsqueeze(-1)
            
        if self.activation:
            y = self.activation(z)
        else:
            y = z
            
        return y, x_grad_out
    
    def forward_nograd(self, x):
        
        z = x.matmul(self.weight.t()) + self.bias
    
        if self.activation:
            y = self.activation(z)
        else:
            y = z
            
        return y