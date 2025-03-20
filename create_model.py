
import torch

from two_stage_models.basic_twostage_separate import BasicTwoStageSeparate, NNTwoStageSeparate


def create_two_stage_model(x_dim:int, z_dim:int, num_classes:int, two_stage_model_name):
    if two_stage_model_name == 'linear':
        two_stage_model = BasicTwoStageSeparate(x_dim, z_dim, num_classes)
    elif two_stage_model_name == 'NN':
        two_stage_model = NNTwoStageSeparate(x_dim, z_dim, num_classes)
    # torch.nn.init.xavier_uniform(two_stage_model.weight)
    return two_stage_model