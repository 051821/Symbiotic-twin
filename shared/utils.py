import torch


def get_device():
    """
    Automatically select GPU if available
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_weights(weight_list, weight_factors):
    """
    Perform weighted average of model weights (FedAvg)

    weight_list: list of state_dicts
    weight_factors: list of float weights (w_i)

    Returns averaged state_dict
    """

    averaged_weights = {}

    for key in weight_list[0].keys():
        averaged_weights[key] = sum(
            weight_factors[i] * weight_list[i][key]
            for i in range(len(weight_list))
        )

    return averaged_weights
