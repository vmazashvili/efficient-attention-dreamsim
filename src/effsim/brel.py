"""B-rel: the released DreamSim dino_vitb16 checkpoint behind our embed_fn interface (SPEC §7.1)."""


def load_brel(device, cache_dir="./models"):
    """embed_fn for unnormalised [0, 1] inputs; the package applies ImageNet normalisation itself.

    normalize_embeds=False keeps D = 1 - cosine as in DreamSim's training and evaluation code;
    the factory's default (True) centres each embedding first, which changes the distance.
    """
    from dreamsim import dreamsim

    model, _ = dreamsim(pretrained=True, dreamsim_type="dino_vitb16", normalize_embeds=False,
                        device=device, cache_dir=str(cache_dir))
    return model.embed
