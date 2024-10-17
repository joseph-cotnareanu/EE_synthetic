


from two_stage_models.basic_twostage_separate import BasicTwoStageSeparate


def create_two_stage_model(x_dim:int, z_dim:int, num_classes:int):
    two_stage_model = BasicTwoStageSeparate(x_dim, z_dim, num_classes)
    return two_stage_model