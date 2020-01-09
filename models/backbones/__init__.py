from models.backbones.resnet import resnet50
from models.backbones.encoder8 import Encoder8
from models.backbones.vgg16 import VGG16TensorflowAlign, VGG16P2M, VGG16Recons
from models.backbones.costvolume import MVSNet


def get_backbone(options, freeze_cv=False):
    if options.backbone.startswith("vgg16"):
        if options.align_with_tensorflow:
            nn_encoder = VGG16TensorflowAlign()
        else:
            nn_encoder = VGG16P2M(pretrained="pretrained" in options.backbone)
        nn_decoder = VGG16Recons()
    elif options.backbone == "resnet50":
        nn_encoder = resnet50()
        nn_decoder = None
    elif options.backbone == "encoder8":
        nn_encoder = Encoder8()
        nn_decoder = None
    elif options.backbone == "costvolume":
        nn_encoder = MVSNet(freeze_cv = freeze_cv)
        nn_decoder = None
    else:
        raise NotImplementedError("No implemented backbone called '%s' found" % options.backbone)
    return nn_encoder, nn_decoder
