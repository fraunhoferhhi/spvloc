import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

from .anchors import Anchors

# from .equi_conv import EquiConv2d, replace_conv_with_equiconv


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels, desc_channels):
        super(EmbeddingModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, desc_channels)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MLPEmbeddingModule(nn.Module):
    def __init__(self, in_channels, desc_channels):
        super(MLPEmbeddingModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, desc_channels),
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class SpatialEmbeddingModule(nn.Module):
    def __init__(self, in_channels, desc_length):
        super(SpatialEmbeddingModule, self).__init__()
        self.embedding_conv = nn.Conv2d(in_channels, desc_length, 1)
        self.embedding_norm = nn.BatchNorm2d(desc_length)

    def forward(self, x):
        x = self.embedding_conv(x)
        x = F.elu(x)
        x = self.embedding_norm(x)
        return x


class ImageModule(nn.Module):
    def __init__(self, config):
        super(ImageModule, self).__init__()

        desc_length = config.MODEL.DESC_LENGTH
        self.desc_length_half = desc_length / 2
        self.normalise = config.MODEL.NORMALISE_EMBEDDING

        self.dim_reduction = False
        backbone = config.MODEL.IMAGE_BACKBONE
        if "efficient_s" in backbone:
            backbone = "efficient_s"

        net, out_dim = _create_backbone(backbone, config.TEST.IGNORE_PRETRAINED)
        out_dim_xf = 640
        self.skip_connection = config.MODEL.IMAGE_BACKBONE == "efficient_s_skip"

        if backbone == "efficient_s":
            final_conv_channels = 1344
            if self.skip_connection:
                self.layers_1a = nn.Sequential(*list(net.children())[0][:2])
                self.layers_1b = nn.Sequential(*list(net.children())[0][2:3])
                self.layers_1c = nn.Sequential(*list(net.children())[0][3:4])
                self.layers_1d = nn.Sequential(*list(net.children())[0][4:5])
                self.layers = nn.Sequential(*list(net.children())[0][5:])
            else:
                self.layers_1 = nn.Sequential(*list(net.children())[0][:4])
                self.layers = nn.Sequential(*list(net.children())[0][4:])
        else:
            final_conv_channels = 640
            self.layers_1 = nn.Sequential(*list(net.children())[:6])
            self.layers = nn.Sequential(*list(net.children())[6:-2])

        self.dim_reduction = final_conv_channels != out_dim_xf

        if self.dim_reduction:
            self.dim_adaption = nn.Conv2d(final_conv_channels, out_dim_xf, 1)

        if config.MODEL.EMBEDDER_TYPE_IMAGE_MODULE == "fc":
            self.embedding = EmbeddingModule(out_dim, desc_length)
        elif config.MODEL.EMBEDDER_TYPE_IMAGE_MODULE == "mlp":
            self.embedding = MLPEmbeddingModule(out_dim, desc_length)
        elif config.MODEL.EMBEDDER_TYPE_IMAGE_MODULE == "spatial":
            self.embedding = SpatialEmbeddingModule(out_dim, desc_length)
            self.normalise = False

    def forward(self, x):
        if self.skip_connection:
            x1a = self.layers_1a(x)  # half resolution
            x1b = self.layers_1b(x1a)  # quater resolution
            x1 = self.layers_1c(x1b)  # eights resolution
            x2a = self.layers_1d(x1)
            x2 = self.layers(x2a)
        else:
            x1 = self.layers_1(x)
            x2 = self.layers(x1)

        x = self.embedding(x2)

        # scale down the features of previous stage to allow concatenation with higher dimensional features
        x1_down = F.interpolate(x1, size=(7, 7), mode="bilinear")
        x2_down = F.interpolate(x2, size=(7, 7), mode="bilinear")
        xf = torch.cat([x2_down, x1_down], dim=1)

        if self.dim_reduction:
            xf = self.dim_adaption(xf)

        if self.normalise:
            x = F.normalize(x)

        if self.skip_connection:
            return (x, x2a, x1, x1b, x1a), xf
        else:
            return x, xf


class PanoFeatExtract(nn.Module):
    def __init__(self, config):
        super(PanoFeatExtract, self).__init__()

        net = models.densenet121(
            weights=None if config.TEST.IGNORE_PRETRAINED else "DenseNet121_Weights.IMAGENET1K_V1"
        ).features
        net.transition3.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

        output_dim = 640

        num_classes = (
            (3 * config.MODEL.PANO_ENCODE_NORMALS)
            + config.MODEL.PANO_ENCODE_SEMANTICS
            + config.MODEL.PANO_ENCODE_DEPTH
        )

        if num_classes == 0:
            raise ValueError("Can not construct PanoFeatExtract with zero input channels")

        if config.MODEL.PANO_USE_EQUICONV:
            raise NotImplementedError("The 'PANO_USE_EQUICONV' feature is currently unavailable.")

        # if config.MODEL.PANO_USE_EQUICONV:
        #    replace_conv_with_equiconv(net)

        modules_dense = list(net.children())

        # change number of input channels
        # if config.MODEL.PANO_USE_EQUICONV:
        #    modules_dense[0] = EquiConv2d(num_classes, 64, kernel_size=7, stride=2, padding=3)
        # else:
        #    modules_dense[0] = torch.nn.Conv2d(num_classes, 64, kernel_size=7, stride=2, padding=3)

        modules_dense[0] = torch.nn.Conv2d(num_classes, 64, kernel_size=7, stride=2, padding=3)
        self.backdense_0 = torch.nn.Sequential(*modules_dense[:1])
        self.backdense_1 = torch.nn.Sequential(*modules_dense[1:])

        self.c1 = nn.Conv2d(1024, output_dim, 1)
        self.n1 = nn.BatchNorm2d(output_dim)

    def forward(self, image):
        x0 = self.backdense_0(image)
        x1 = self.backdense_1(x0)
        xf = self.n1(F.elu(self.c1(x1)))
        return xf


class LayoutDecoder_V2(nn.Module):
    def __init__(self, config):
        super().__init__()
        desc_length = config.MODEL.DESC_LENGTH
        layout_channels = 3

        fm = 1  # feature multiplicator, increases size of network
        semantic_classes = config.MODEL.DECODER_SEMANTIC_CLASSES
        decoder_resolution = config.MODEL.DECODER_RESOLUTION

        self.tight_bottleneck = config.MODEL.EMBEDDER_TYPE_IMAGE_MODULE != "spatial"
        self.skip_connection = config.MODEL.IMAGE_BACKBONE == "efficient_s_skip"

        if self.tight_bottleneck:
            self.fc = nn.Sequential(nn.Linear(desc_length, 2048), nn.ReLU())
            decorder_input_dim = 2048
        else:
            decorder_input_dim = desc_length

        if not self.skip_connection:
            upsample_layers = [
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
                nn.Conv2d(decorder_input_dim, 128 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
                nn.Conv2d(128 * fm, 64 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
                nn.Conv2d(64 * fm, 32 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
                nn.Conv2d(32 * fm, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Upsample(size=decoder_resolution, mode="bilinear"),
            ]
            self.decov_layers = nn.Sequential(*upsample_layers)
        else:
            self.upsample_layer = nn.Upsample(scale_factor=2.0, mode="bilinear")

            self.block_1 = nn.Sequential(
                nn.Conv2d(decorder_input_dim + 128, 128 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
            )

            self.block_2 = nn.Sequential(
                nn.Conv2d(128 * fm + 64, 64 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
            )

            self.block_3 = nn.Sequential(
                nn.Conv2d(64 * fm + 48, 32 * fm, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32 * fm),
                nn.Upsample(scale_factor=2.0, mode="bilinear"),
            )

            self.block_4 = nn.Sequential(
                nn.Conv2d(32 * fm, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Upsample(size=decoder_resolution, mode="bilinear"),
            )

        semantic_head_layers = [
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, semantic_classes, 1),
        ]

        layout_head_layers = [
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, layout_channels, 1),
        ]

        layout_head_layers.append(nn.Tanh())
        self.semantic_head = nn.Sequential(*semantic_head_layers)
        self.layout_head = nn.Sequential(*layout_head_layers)

    def forward(self, x):
        if self.tight_bottleneck:
            x = self.fc(x)

            # NOTE: For square images x.view(-1, 128, 4, 4) could be more meaningful.
            # The (in-channels have to be adapted accordningly), non-square images could be padded.
            x = x.view(-1, 256, 2, 4)

        if not self.skip_connection:
            x = self.decov_layers(x)
            x_s = self.semantic_head(x)
            x_l = self.layout_head(x)
        else:
            x_ = self.upsample_layer(x[0])
            x_ = self.block_1(torch.concat([x_, x[1]], axis=1))
            x_ = self.block_2(torch.concat([x_, x[2]], axis=1))
            x_ = self.block_3(torch.concat([x_, x[3]], axis=1))
            x_ = self.block_4(x_)
            x_s = self.semantic_head(x_)
            x_l = self.layout_head(x_)

        x = torch.cat([x_l, x_s], dim=1)

        return x


class PoseEstimatorMLP(nn.Module):
    def __init__(self, n=640):
        super(PoseEstimatorMLP, self).__init__()

        self.fc1 = nn.Linear(n, 1280)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1280, 640)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(640, 128)
        self.relu3 = nn.ReLU()

        # translation branch
        self.fc4_1 = nn.Linear(128, 3)

        # rotation branch
        self.fc4_2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))

        output1 = self.fc4_1(x)
        output2 = self.fc4_2(x)

        return output1, output2


class FeatureCompression(nn.Module):
    def __init__(self, num_features_in, feature_size=320, feature_size_2=160, feature_size_3=80):
        super(FeatureCompression, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(feature_size, feature_size_2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(feature_size_2, feature_size_3, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(2, stride=2)
        self.act3 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.act2(out)
        out = self.conv3(out)
        # out = self.pool2(out)
        out = self.act3(out)
        return out.contiguous().view(x.shape[0], -1)


# BBoxTransform, ClipBoxes, ClassificationModel, RegressionModel are adapted and modified from:
# https://github.com/yhenon/pytorch-retinanet/blob/0348a9d57b279e3b5b235461b472d37da5feec3d/retinanet/utils.py
# Matching is adapted and modified from Dtoid: Deep Template-based Object Instance Detection by Mercier et al.


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        mean = self.mean.to(boxes.device)
        std = self.std.to(boxes.device)

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * std[0] + mean[0]
        dy = deltas[:, :, 1] * std[1] + mean[1]
        dw = deltas[:, :, 2] * std[2] + mean[2]

        if deltas.size(2) == 3:
            dh = deltas[:, :, 2] * std[3] + mean[3]
        else:
            dh = deltas[:, :, 3] * std[3] + mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()
        self.width = width
        self.height = height

    def forward(self, boxes, img=None):
        if img is None:
            width = self.width
            height = self.height
        else:
            batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ELU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ELU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        out1 = out.permute(0, 2, 3, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes), out


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ELU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ELU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ELU()
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class BoundingBoxDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_size = config.RENDER.PANO_SIZE
        img_out_size = config.RENDER.PANO_SIZE
        self.correlation_model = CorrelationModel(
            img_out_size,
            input_dim=640,
            correlation_dim=640,
            size_extension_factor=config.MODEL.PANO_BB_EXTENSION_FACTOR,
        )

        self.anchors = Anchors(
            pyramid_levels=[4], ratios=[0.4, 0.6, 0.8, 1, 1.2], sizes=[20], scales=[1, 2, 3, 4, 5, 6, 7]
        )

        self.classification = ClassificationModel(512, num_anchors=35)
        self.regression = RegressionModel(512, num_anchors=35)

        # weight init
        prior = 0.01
        self.classification.output.weight.data.fill_(0)
        self.classification.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regression.output.weight.data.fill_(0)
        self.regression.output.bias.data.fill_(0)

        self.correlation_model.mask_decoder.seg_out.weight.data.fill_(0)
        self.correlation_model.mask_decoder.seg_out.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes(height=img_size[0], width=img_size[1])

    def forward(self, pano_feat, persp_feat, test=False, ignore_regression=False):
        xcors, segmentation = self.correlation_model(pano_feat, persp_feat, return_vp_mask=test)

        anchors = self.anchors([[xcors.size(2), xcors.size(3)]])
        classifications, _ = self.classification(xcors)
        if ignore_regression:
            regression = None
        else:
            regression = self.regression(xcors)

        return classifications, regression, anchors, segmentation, xcors

    def forward_best_anchors(self, pano_feat, persp_feat, topk=128):
        bs = pano_feat.shape[0]

        classifications, regression, anchors, segmentation, xcors = self.forward(pano_feat, persp_feat)

        transformed_anchors = self.regressBoxes.forward(anchors, regression)

        classifications_ = classifications.contiguous().view(bs, -1, 2)

        transformed_anchors = transformed_anchors.view(bs, -1, 4)

        maxes = torch.topk(classifications_, topk, dim=1)
        max_score = maxes[0][:, :, 1]
        max_id = maxes[1][:, :, 1]
        anchors_pred = torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(transformed_anchors, max_id)]
        )
        best_anchors = torch.cat((anchors_pred[:, 0:topk], torch.unsqueeze(max_score[:, 0:topk], 2)), dim=2)
        best_anchors = torch.flatten(best_anchors, start_dim=1)

        return classifications, regression, anchors, segmentation, best_anchors, xcors

    def forward_score_and_corr(self, pano_feat, persp_feat, test=False):
        bs = pano_feat.shape[0]

        classifications, _, _, _, xcors = self.forward(pano_feat, persp_feat, test=test, ignore_regression=True)

        classifications_ = classifications.contiguous().view(bs, -1, 2)

        maxes = torch.topk(classifications_, 1, dim=1)
        max_score = maxes[0][:, :, 1]
        return max_score.squeeze(dim=1), xcors

    def forward_xcors_only(self, pano_feat, persp_feat):
        xcors, _ = self.correlation_model(pano_feat, persp_feat, test=True)
        return xcors

    def forward_regress_boxes(self, pano_feat, persp_feat, topk=1, test=False):
        bs = pano_feat.shape[0]

        classifications, regression, anchors, segmentation, xcors = self.forward(pano_feat, persp_feat, test=test)

        transformed_anchors = self.regressBoxes.forward(anchors, regression)

        classifications = classifications.contiguous().view(bs, -1, 2)
        regression = regression.contiguous().view(bs, -1, 2)
        transformed_anchors = transformed_anchors.view(bs, -1, 4)

        maxes = torch.topk(classifications, 1000, dim=1)
        max_score = maxes[0][:, :, 1]
        max_id = maxes[1][:, :, 1]
        anchors_pred = torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(transformed_anchors, max_id)]
        )

        best_anchors = torch.cat((anchors_pred[:, 0:128], torch.unsqueeze(max_score[:, 0:128], 2)), dim=2)
        best_anchors = torch.flatten(best_anchors, start_dim=1)

        max_score_out = []
        anchors_pred_out = []
        for i in range(bs):
            nms_ids = torchvision.ops.boxes.nms(anchors_pred[i], max_score[i], 0.5)
            max_score_temp = max_score[i, nms_ids]
            anchors_pred_temp = anchors_pred[i, nms_ids]
            max_score_out.append(max_score_temp[:topk])
            anchors_pred_out.append(anchors_pred_temp[:topk])
        max_score_out = torch.cat(max_score_out)
        anchors_pred_out = torch.cat(anchors_pred_out)

        return [max_score_out, anchors_pred_out, segmentation, best_anchors, xcors]


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, upsampling_factor=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_dim)
        self.upsampling_factor = upsampling_factor

    def forward(self, x):
        x = self.bn(F.elu(self.conv(x)))
        if self.upsampling_factor:
            x = F.interpolate(x, scale_factor=self.upsampling_factor)
        return x


class ViewportDecoder(nn.Module):
    def __init__(self, img_size):
        super(ViewportDecoder, self).__init__()
        self.block1 = ConvBlock(512, 256, upsampling_factor=2)
        self.block2 = ConvBlock(256, 128, upsampling_factor=2)
        self.block3 = ConvBlock(128, 64, upsampling_factor=2)
        self.block4 = ConvBlock(64, 32)
        self.block5 = ConvBlock(32, 16)
        self.seg_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.img_size = img_size

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.interpolate(x, size=self.img_size)
        x = self.block5(x)
        return self.seg_out(x)


class CorrelationModel(nn.Module):
    def __init__(self, img_size, input_dim=1024, correlation_dim=256, size_extension_factor=1.0):
        super(CorrelationModel, self).__init__()

        self.f_conv1 = ConvBlock(input_dim, correlation_dim, padding=0)
        self.f_conv2 = ConvBlock(correlation_dim, correlation_dim, padding=0)

        self.r1_conv = ConvBlock(correlation_dim, 256)
        self.r2_conv = ConvBlock(correlation_dim, 256)
        self.r3_conv = ConvBlock(correlation_dim, 256)

        self.r_star_conv = ConvBlock(768, 512)
        self.mask_decoder = ViewportDecoder(img_size)
        self.scaling = size_extension_factor

    def forward(self, pano_features, image_features, return_vp_mask=False):
        f = self.f_conv1(image_features)
        f_hat = self.f_conv2(f)  # F^

        f_tilde = F.avg_pool2d(image_features, 7)  # F~

        r1 = self.r1_conv(conv2d_dw_group(pano_features, f_hat, padding=1))
        r2 = self.r2_conv(pano_features * f_tilde)
        r3 = self.r3_conv(pano_features - f_tilde)

        r_star = self.r_star_conv(torch.cat([r2, r3, r1], dim=1))  # R*

        if not return_vp_mask:
            segmentation = self.mask_decoder(r_star)
            if self.scaling > 1.0:
                size_out = (int(r_star.shape[2] * self.scaling), int(r_star.shape[3] * self.scaling))
                r_star = F.interpolate(r_star, size=size_out, mode="bilinear")  # R*

            return r_star, segmentation

        if self.scaling > 1.0:
            size_out = (int(r_star.shape[2] * self.scaling), int(r_star.shape[3] * self.scaling))
            r_star = F.interpolate(r_star, size=size_out, mode="bilinear")  # R*

        return r_star, 0


# Adapted from SiamMask
# https://github.com/foolwood/SiamMask/blob/master/models/rpn.py
def conv2d_dw_group(x, kernel, padding=0):
    batch, channel = kernel.shape[:2]
    x = x.contiguous().view(1, batch * channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, kernel, groups=batch * channel, padding=padding)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def _create_backbone(name, inference_mode=False):
    backbones = {
        "resnet18": (models.resnet18(weights=None if inference_mode else "ResNet18_Weights.IMAGENET1K_V1"), 512),
        "resnet50": (models.resnet50(weights=None if inference_mode else "ResNet50_Weights.IMAGENET1K_V1"), 2048),
        "efficient_s": (
            models.efficientnet_v2_s(weights=None if inference_mode else "EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
            1280,
        ),
    }
    return backbones[name]
