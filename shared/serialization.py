import torch


def serialize_model(model):
    """
    Convert model state_dict to JSON-serializable dictionary
    """
    return {
        key: value.cpu().detach().numpy().tolist()
        for key, value in model.state_dict().items()
    }


def deserialize_model(model, serialized_state):
    """
    Load serialized dictionary back into model
    """
    state_dict = {
        key: torch.tensor(value)
        for key, value in serialized_state.items()
    }

    model.load_state_dict(state_dict)
    return model


def extract_weights(model):
    """
    Returns raw state_dict
    """
    return model.state_dict()


def load_weights(model, weights):
    """
    Loads weights into model
    """
    model.load_state_dict(weights)
    return model
