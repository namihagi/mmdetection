import torch
from torch import nn
from mmcv.ops import nms

from ..builder import RAND_BOX


@RAND_BOX.register_module()
class RandBox(nn.Module):
    """Random box generation for contrastive learning.

    This module generates random boxes instead of gt_boxes.
    """

    def __init__(self,
                 flip=True,
                 nms_thr=0.7,
                 num_of_init_boxes=2000,
                 min_scale_rate=0.1,
                 min_num_of_final_box=5,
                 max_num_of_final_box=50,
                 box_augment=False,
                 box_augment_scale=0.3):
        super(RandBox, self).__init__()

        self.flip = flip
        self.nms_thr = nms_thr
        self.with_nms = self.nms_thr is not None
        self.num_of_init_boxes = num_of_init_boxes
        self.min_scale_rate = min_scale_rate
        self.min_num_of_final_box = min_num_of_final_box
        self.max_num_of_final_box = max_num_of_final_box
        self.box_augment = box_augment
        self.box_augment_scale = box_augment_scale
        assert 0 < self.box_augment_scale < 0.5, \
            "self.box_augment_scale need to in range [0, 0.5]."

    def forward(self, img, im_metas):
        """Forward features from the upstream network.

        Args:
            img (Tensor): of shape (N, C * 2, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
            rand_box_1 (list[Tensor]): random boxes like gt_bboxes
                in [lt_x, lt_y, rb_x, rb_y] format.

            rand_box_2 (list[Tensor]): if self.flip is True,
                this is horizontal flipped rand_box_1.

            img_1 (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_2 (Tensor): of shape (N, C, H, W) encoding input images.
                if self.flip is True, horizontal flipped.
        """

        device = img.device
        num_img = img.size(0)

        with torch.no_grad():
            img_1 = img[:, :3, :, :]
            img_2 = img[:, 3:, :, :]

            rand_box_1 = self.generate_rand_box(im_metas, device, num_img)

            if self.flip:
                rand_box_2 = []
                for i in range(num_img):
                    boxes_1 = rand_box_1[i]
                    shape = im_metas[i]['img_shape']

                    boxes_2 = boxes_1.clone()
                    boxes_2[:, 0] = shape[1] - (boxes_1.clone()[:, 2] + 1)
                    boxes_2[:, 2] = shape[1] - (boxes_1.clone()[:, 0] + 1)
                    rand_box_2.append(boxes_2)

                    img_2[i, :, :shape[0], :shape[1]] = \
                        img_2[i, :, :shape[0], :shape[1]].flip(dims=(-1,))
            else:
                rand_box_2 = rand_box_1

            if self.box_augment:
                rand_box_1 = self.augment_bbox(rand_box_1, num_img,
                                               device, im_metas)
                rand_box_2 = self.augment_bbox(rand_box_2, num_img,
                                               device, im_metas)

        return rand_box_1, rand_box_2, img_1, img_2

    def generate_rand_box(self, im_metas, device, num_img):
        rand_boxes_init = \
            torch.rand(num_img, self.num_of_init_boxes, 4).to(device)
        pre_rand_boxes = torch.zeros_like(rand_boxes_init).to(device)

        # pre_rand_boxes[:, :, 0] (tl_x) < pre_rand_boxes[:, :, 2] (br_x)
        pre_rand_boxes[:, :, 0] = \
            torch.where(rand_boxes_init[:, :, 0] < rand_boxes_init[:, :, 2],
                        rand_boxes_init[:, :, 0], rand_boxes_init[:, :, 2])
        pre_rand_boxes[:, :, 2] = \
            torch.where(rand_boxes_init[:, :, 0] < rand_boxes_init[:, :, 2],
                        rand_boxes_init[:, :, 2], rand_boxes_init[:, :, 0])

        # pre_rand_boxes[:, :, 1] (tl_y) < pre_rand_boxes[:, :, 3] (br_y)
        pre_rand_boxes[:, :, 1] = \
            torch.where(rand_boxes_init[:, :, 1] < rand_boxes_init[:, :, 3],
                        rand_boxes_init[:, :, 1], rand_boxes_init[:, :, 3])
        pre_rand_boxes[:, :, 3] = \
            torch.where(rand_boxes_init[:, :, 1] < rand_boxes_init[:, :, 3],
                        rand_boxes_init[:, :, 3], rand_boxes_init[:, :, 1])

        # pseudo batch_scores
        if self.with_nms:
            pseudo_scores = \
                torch.rand(num_img, self.num_of_init_boxes, 1).to(device)

        num_of_boxes_per_img = torch.randint(self.min_num_of_final_box,
                                             self.max_num_of_final_box,
                                             size=(num_img,)).to(device)

        rand_boxes = []
        for i in range(num_img):
            shape = im_metas[i]['img_shape']
            pre_rand_box = pre_rand_boxes[i]
            if self.with_nms:
                pseudo_score = pseudo_scores[i]
            num_of_final_boxes = num_of_boxes_per_img[i]

            # rescale length of box
            pre_rand_box[:, 0::2] *= shape[1]  # width
            pre_rand_box[:, 1::2] *= shape[0]  # height

            # get the min length of box
            min_height = shape[0] * self.min_scale_rate
            min_width = shape[1] * self.min_scale_rate

            # calculate box height and width
            box_height = pre_rand_box[:, 3] - pre_rand_box[:, 1]
            box_width = pre_rand_box[:, 2] - pre_rand_box[:, 0]

            # get indices larger than threshold
            box_keep_idx = (box_height > min_height) * (box_width > min_width)

            # get boxes and scores to keep
            pre_rand_box_keep = pre_rand_box[box_keep_idx]
            if self.with_nms:
                pseudo_score_keep = pseudo_score[box_keep_idx]

            if self.with_nms:
                # sort boxes and scores about scores
                sorted_psuedo_score, order = \
                    torch.sort(pseudo_score_keep, 0, True)
                sorted_pre_rand_box_keep = pre_rand_box_keep[order]

                # nms
                # dets's shape: [num_box, 5] in [lt_x, lt_y, rb_x, rb_y, score]
                dets, _ = nms(sorted_pre_rand_box_keep.view(-1, 4),
                              sorted_psuedo_score.view(-1),
                              self.nms_thr)
            else:
                dets = pre_rand_box_keep.view(-1, 4)

            # take topN from boxes after nms
            if dets.size(0) < num_of_final_boxes:
                rand_boxes.append(dets[:, :4])
            else:
                rand_boxes.append(dets[:num_of_final_boxes, :4])

        return rand_boxes

    def augment_bbox(self, bboxes_list, num_img, device, im_metas):

        for i in range(num_img):
            bboxes = bboxes_list[i]
            shape = im_metas[i]['img_shape']
            num_box = bboxes.size(0)

            base_width = \
                (bboxes[:, 2] - bboxes[:, 0]) * self.box_augment_scale
            base_height = \
                (bboxes[:, 3] - bboxes[:, 1]) * self.box_augment_scale

            rand_offset = torch.rand(num_box, 4).to(device)
            rand_offset = (rand_offset * 2) - 1.0

            offset_width = \
                base_width.view(-1, 1) * rand_offset[:, 0::2]
            offset_height = \
                base_height.view(-1, 1) * rand_offset[:, 1::2]

            bboxes[:, 0::2] += offset_width
            bboxes[:, 1::2] += offset_height

            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, shape[1])
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, shape[0])

        return bboxes_list


if __name__ == "__main__":
    rand_box = RandBox()
    gt_boxes = torch.tensor([[536.1965, 543.7751, 735.5865, 800.0000],
                             [1.7610, 31.8126, 580.5970, 785.0679],
                             [141.9691, 229.6019, 1105.5155, 800.0000],
                             [1013.2112, 380.5714, 1198.5879, 800.0000],
                             [0.0000, 416.4871, 91.6672, 546.0422],
                             [154.5586, 422.4637, 515.8885, 792.8057],
                             [1024.6953, 381.8642, 1188.6960, 577.3302]])
    gt_boxes = [gt_boxes, gt_boxes, gt_boxes]
    im_metas = {'img_shape': (800, 1199, 3)}
    im_metas = [im_metas, im_metas, im_metas]
    rand_boxes = rand_box(gt_boxes, im_metas)
    for i in range(len(rand_boxes)):
        print(rand_boxes[i].shape)
