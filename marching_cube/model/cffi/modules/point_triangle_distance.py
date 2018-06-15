from torch.nn.modules.module import Module
from functions.point_triangle_distance import DistanceFunction 

class DistanceModule(Module):
    def forward(self, input1, input2):
        dis = DistanceFunction()(input1, input2)
        return dis 
