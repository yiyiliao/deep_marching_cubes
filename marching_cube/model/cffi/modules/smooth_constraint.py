from torch.nn.modules.module import Module
from functions.smooth_constraint import CurvatureFunction, OccupancyConnectivity 

class CurvatureModule(Module):
    def forward(self, offset, topology):
        loss = CurvatureFunction()(offset, topology)
        return loss

class OccupancyModule(Module):
    def forward(self, occupancy):
        loss = OccupancyConnectivity()(occupancy)
        return loss
