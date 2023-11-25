from fastMONAI.vision_all import ZNormalization, PadOrCrop, RandomAffine, RandomNoise, RandomGamma

def get_transforms(logger, augmentation, size):
    logger.info('Adding augmentations')
    random_affine = RandomAffine(degrees=15, translation=15)
    random_gamma = RandomGamma()
    random_noise = RandomNoise()

    item_tfms = [ZNormalization(), PadOrCrop(size)]

    if augmentation == 'rand_affine':
        logger.info('Random Affine is added')
        item_tfms.append(random_affine)
    elif augmentation == 'rand_gamma':
        logger.info('Random Gamma is added')
        item_tfms.append(random_gamma)
    elif augmentation == 'rand_noise':
        logger.info('Random Noise is added')
        item_tfms.append(random_noise)
    else:
        pass

    return item_tfms
