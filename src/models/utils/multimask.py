# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, actions=None):            
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            # 检查backbone是否接受actions参数
            if hasattr(self.backbone, 'use_dynamics') and self.backbone.use_dynamics and actions is not None:
                outs += [self.backbone(zi, hi, mc, mt, actions=actions, mask_index=i)]
            else:
                # 如果不接受actions参数，则不传递
                outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs
