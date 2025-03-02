from fvcore.common.registry import Registry

from common.type_utils import cfg2dict


VISION_REGISTRY = Registry("vision")
LANGUAGE_REGISTRY = Registry("language")
GROUNDING_REGISTRY = Registry("grounding")
HEADS_REGISTRY = Registry("heads")


def build_module(module_type, cfg):
    if module_type == "vision":
        return VISION_REGISTRY.get(cfg.name)(cfg, **cfg2dict(cfg.args))
    elif module_type == "language":
        return LANGUAGE_REGISTRY.get(cfg.name)(cfg, **cfg2dict(cfg.args))
    elif module_type == "grounding":
        return GROUNDING_REGISTRY.get(cfg.name)(cfg, **cfg2dict(cfg.args))
    elif module_type == "heads":
        return HEADS_REGISTRY.get(cfg.name)(cfg, **cfg2dict(cfg.args))
    else:
        raise NotImplementedError(f"module type {module_type} not implemented")