import torch
import torch.nn as nn


def circular_distance_loss(estimated_rotations, gt_rotations):
    # Normalize the estimated and ground truth rotations to [-pi, pi]
    estimated_rotations_normalized = (estimated_rotations + torch.pi) % (2 * torch.pi) - torch.pi
    gt_rotations_normalized = (gt_rotations + torch.pi) % (2 * torch.pi) - torch.pi

    # Calculate the difference
    diff = estimated_rotations_normalized - gt_rotations_normalized

    # Take the absolute value to ensure positive distance
    diff_abs = torch.abs(diff)

    # If the result is greater than pi, use the shortest distance
    mask = diff_abs > torch.pi
    diff_abs[mask] = 2 * torch.pi - diff_abs[mask]

    # Average the circular distances across the batch
    loss = torch.mean(diff_abs)

    return loss


def cross_entropy_loss_masked(logits, labels, mask):
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Calculate the raw loss
    loss = criterion(logits, labels)

    mask = mask.squeeze(1)

    # Check if the mask is entirely zero
    # if mask.sum() == 0:
    #    return torch.tensor(0.0)

    # Apply the mask to the loss
    masked_loss = loss * mask

    # Compute the mean loss, considering only the unmasked elements
    loss = masked_loss.sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1.0)

    return loss.mean()


def cosine_similarity_loss_masked(decoded_normals, target_normals, mask):
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    # Compute the cosine similarity
    similarity = cosine_similarity(decoded_normals, target_normals)
    similarity = (similarity + 1.0) / 2.0  # Adjusting the range to 0 to 1

    similarity = similarity * mask.squeeze(1)

    # Calculate the loss
    loss = torch.sum(similarity, [1, 2])
    loss = 1.0 - (loss / (mask.sum(dim=(1, 2, 3)) + 1.0))

    return loss.mean()


def cosine_similarity_loss(decoded_normals, target_normals):
    # loss = F.cosine_embedding_loss(decoded_normals, target_normals, dim=-1) # use only a single image
    loss = torch.nn.CosineSimilarity(dim=1)(decoded_normals, target_normals)

    loss = torch.mean(loss, [1, 2])
    loss = 1.0 - ((loss + 1.0) / 2.0)

    return loss.mean()


# Adapted and modified from:
# https://github.com/yhenon/pytorch-retinanet/blob/39938d285fc3c602ae6a697ff690bbca29a502be/retinanet/losses.py
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


# Adapted and modified from:
# https://github.com/yhenon/pytorch-retinanet/blob/39938d285fc3c602ae6a697ff690bbca29a502be/retinanet/losses.py
class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1.0 - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)
            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(
            dim=0, keepdim=True
        )


class MultiLossCriterion(nn.Module):
    def __init__(
        self,
        t_loss_fn=nn.L1Loss(),
        q_loss_fn=nn.L1Loss(),
        sax=0,
        saq=0,
        sa_l2=0,
        sa_classification=0,
        sa_regression=0,
        sa_mask=0,
        sa_decode_layout=0,
        sa_decode_semantics=0,
        learn_beta=True,
    ):
        super(MultiLossCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.sax_2 = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.sax_3 = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)

        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.sa_l2 = nn.Parameter(torch.Tensor([sa_l2]), requires_grad=learn_beta)
        self.sa_classification = nn.Parameter(torch.Tensor([sa_classification]), requires_grad=learn_beta)
        self.sa_regression = nn.Parameter(torch.Tensor([sa_regression]), requires_grad=learn_beta)
        self.sa_mask = nn.Parameter(torch.Tensor([sa_mask]), requires_grad=learn_beta)
        self.sa_decode_layout = nn.Parameter(torch.Tensor([sa_decode_layout]), requires_grad=learn_beta)
        self.sa_decode_semantics = nn.Parameter(torch.Tensor([sa_decode_semantics]), requires_grad=learn_beta)

    def forward(
        self,
        pred_abs_T,
        target_abs_T,
        pred_R,
        targ_R,
        classification_loss,
        regression_loss,
        mask_bce_losss,
        loss_decode_layout=torch.tensor(0.0),
        loss_decode_semantics=torch.tensor(0.0),
    ):
        s = pred_abs_T.size()

        if len(s) == 3:
            pred_abs_T = pred_abs_T.contiguous().view(-1, *s[2:])
            target_abs_T = target_abs_T.contiguous().view(-1, *s[2:])

        t_loss = torch.tensor(0.0)
        t_loss_3 = torch.tensor(0.0)

        t_loss_2 = self.t_loss_fn(pred_abs_T, target_abs_T)

        q_loss = self.q_loss_fn(pred_R, targ_R)

        sax_exp = torch.exp(-self.sax)
        sax_exp_2 = torch.exp(-self.sax_2)
        sax_exp_3 = torch.exp(-self.sax_3)
        saq_exp = torch.exp(-self.saq)

        loss = (
            sax_exp * t_loss
            + self.sax
            + sax_exp_2 * t_loss_2
            + self.sax_2
            + sax_exp_3 * t_loss_3
            + self.sax_3
            + saq_exp * q_loss
            + self.saq
        )
        loss += self.forward_no_pose(
            classification_loss,
            regression_loss,
            mask_bce_losss,
            loss_decode_layout,
            loss_decode_semantics,
        )
        return loss, t_loss, q_loss, t_loss_2, t_loss_3

    def forward_no_pose(
        self,
        classification_loss,
        regression_loss,
        mask_bce_losss,
        loss_decode_layout=torch.tensor(0.0),
        loss_decode_semantics=torch.tensor(0.0),
    ):
        l2_loss = torch.tensor(0.0)
        sa_l2_exp = torch.exp(-self.sa_l2)
        sa_classification_exp = torch.exp(-self.sa_classification)
        sa_regression_exp = torch.exp(-self.sa_regression)
        sa_mask_exp = torch.exp(-self.sa_mask)
        sa_decode_layout_exp = torch.exp(-self.sa_decode_layout)
        sa_decode_semantic_exp = torch.exp(-self.sa_decode_semantics)

        loss = (
            sa_l2_exp * l2_loss
            + self.sa_l2
            + sa_classification_exp * classification_loss
            + self.sa_classification
            + sa_regression_exp * regression_loss
            + self.sa_regression
            + sa_mask_exp * mask_bce_losss
            + self.sa_mask
        )
        if loss_decode_layout != torch.tensor(0.0):
            loss += sa_decode_layout_exp * loss_decode_layout + self.sa_decode_layout
        if loss_decode_semantics != torch.tensor(0.0):
            loss += sa_decode_semantic_exp * loss_decode_semantics + self.sa_decode_semantics

        return loss
