import torchvision.transforms as transforms


def build_transform(config, is_train=False, is_layout=False, skip_pil=False, augment=True, pad_height=0):
    antialias = None if is_layout else True

    transform = [
        transforms.Resize(config.INPUT.IMG_SIZE, antialias=antialias),
    ]

    if not is_layout and augment:
        transform += [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5)], p=0.33),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.3)], p=0.33),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.3)], p=0.33),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ]
    transform += [transforms.Pad(padding=(0, pad_height), fill=0), transforms.ToTensor()]

    if is_layout and not skip_pil:
        transform = [
            transforms.ToPILImage(),
        ] + transform
    else:
        transform += [transforms.Normalize(mean=config.INPUT.NORMALISE_MEAN, std=config.INPUT.NORMALISE_STD)]

    transform = transforms.Compose(transform)

    return transform


def denormalize(config, img):
    mean = config.INPUT.NORMALISE_MEAN
    std = config.INPUT.NORMALISE_STD

    denormalize_transform = transforms.Compose(
        [
            transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]),
            transforms.ToPILImage(),
        ]
    )

    return denormalize_transform(img)
