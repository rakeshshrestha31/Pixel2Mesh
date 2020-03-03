import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils

import config

##
#  @param tensor batch x view x height x width tensor
#  @return tensor batch x view x 3 x height x width
def convert_to_3_channel(tensor):
    return tensor.unsqueeze(2).expand(-1, -1, 3, -1, -1)

##
#  @param tensor batch x view x height x width tensor
#  @return tensor batch x view x config.IMG_SIZE x config.IMG_SIZE tensor
def resize_batch(tensor):
    return F.interpolate(
        tensor, size=[config.IMG_SIZE, config.IMG_SIZE], mode='nearest'
    )

##
#  @param tensor batch x view x height x width tensor
#  @return tensor batch x view x 3 x config.IMG_SIZE x config.IMG_SIZE tensor
def process_batch(tensor):
    return  convert_to_3_channel(resize_batch(tensor))

class DepthRenderer(object):

    @staticmethod
    def depth_batch_visualize(batch_input, batch_output, atmost=3):
        batch_size = min(batch_input["depths"].size(0), atmost)
        input_depths = process_batch(batch_input["depths"][:batch_size])
        output_depths = batch_output["depths"] * batch_input["masks"]
        output_depths = process_batch(output_depths[:batch_size])

        concat_depths = torch.cat((input_depths, output_depths), dim=1)
        concat_depths = concat_depths.view(-1, *(concat_depths.size()[2:]))

        # 6 row cuz 3 input and 3 output
        grid_depth = torchvision.utils.make_grid(concat_depths, nrow=6)
        return grid_depth

