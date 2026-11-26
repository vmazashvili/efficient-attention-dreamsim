"""B-rel: the released DreamSim dino_vitb16 checkpoint behind our embed_fn interface (SPEC §7.1)."""


def load_brel(device, cache_dir="./models", normalize_embeds=False):
    """embed_fn for unnormalised [0, 1] inputs; the package applies ImageNet normalisation itself.

    normalize_embeds=False keeps D = 1 - cosine as in DreamSim's training and evaluation code;
    True (the factory's default) centres each embedding first, which changes the distance.
    """
    from dreamsim import dreamsim

    model, _ = dreamsim(pretrained=True, dreamsim_type="dino_vitb16", normalize_embeds=normalize_embeds,
                        device=device, cache_dir=str(cache_dir))
    return model.embed
