from src.data.adapters.llava import LLaVAInstructAdapter


ADAPTER_REGISTRY = {
    "llava_instruct": LLaVAInstructAdapter,
}


def build_adapter(name: str, **kwargs):
    try:
        adapter_cls = ADAPTER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset adapter: {name}. "
            f"Available adapters: {list(ADAPTER_REGISTRY)}"
        )

    return adapter_cls(**kwargs)