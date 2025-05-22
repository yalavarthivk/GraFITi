"""Testing Attribute Serialization."""


from torch import jit, nn


def validate_model_attribute(model: nn.Module, attr: str) -> None:
    r"""Check whether attr is maintained under torhc JIT."""
    original_model = model

    if hasattr(original_model, attr):
        attribute = getattr(original_model, attr)
        print(f"{original_model}.{attr}={attribute}")
    else:
        print(f"{original_model}.{attr} does not exist")

    serialized_model = jit.script(model)

    if hasattr(serialized_model, attr):
        attribute = getattr(serialized_model, attr)
        print(f"{serialized_model}.{attr}={attribute}")
    else:
        print(f"{serialized_model}.{attr} does not exist")

    jit.save(serialized_model, "model.pt")
    loaded_model = jit.load("model.pt")

    if hasattr(loaded_model, attr):
        attribute = getattr(loaded_model, attr)
        print(f"{loaded_model}.{attr}={attribute}")
    else:
        print(f"{loaded_model}.{attr} does not exist")
