import torch


class LinearMemory(torch.nn.Module):

    def __init__(self, dim_subnet, dim_environment, rates, A_SE=None):
        super(LinearMemory, self).__init__()
        self.dim_subnet = dim_subnet
        self.dim_environment = dim_environment
        self.A_EE = torch.nn.Parameter(torch.zeros(dim_environment, dim_environment))
        self.A_ES = torch.nn.Parameter(torch.zeros((dim_environment, dim_subnet)))
        if A_SE is None:
            self.A_SE = torch.nn.Parameter(torch.zeros((dim_subnet, dim_environment)))
        else:
            self.A_SE = A_SE
        self.b = torch.nn.Parameter(torch.ones(dim_environment))
        self.rates = rates

    def forward(self, time, state):
        # split state
        subnet = state[:self.dim_subnet]
        environment = state[self.dim_subnet:]
        # compute contributions for subnet and main
        #print(self.forward_subnet(time, subnet))
        d_subnet = self.forward_subnet(time, subnet) + self.A_SE @ environment
        d_environment = self.A_EE @ environment + self.A_ES @ subnet + self.b
        # concatenate
        dydt = torch.cat([d_subnet, d_environment])
        # dydt = torch.zeros(self.dim_subnet+self.dim_environment)
        # dydt[:self.dim_subnet] = d_subnet
        # dydt[self.dim_subnet:] = d_environment
        return(dydt)

    def forward_subnet(self, time, subnet):
        raise NotImplementedError
