import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .module import *
from .update import BasicUpdateBlockDepth
from functools import partial

Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, pixelwise_net):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shape[1], num_depth)
        batch, feat_channel, height, width = features[0].shape
        dtype, device = features[0].dtype, features[0].device

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        pixelwise_weight_sum = 1e-5 * torch.ones((batch, 1, 1, height, width), dtype=dtype, device=device)
        volume_sum = torch.zeros((batch, feat_channel, num_depth, height, width), dtype=dtype, device=device)

        view_weights_list = []

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_feature = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)
            warped_volume = (warped_feature * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            # calculate pixel-wise view weights
            view_weight = pixelwise_net(warped_volume)
            view_weights_list.append(view_weight)

            if self.training:
                volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
            else:
                volume_sum += warped_volume * view_weight.unsqueeze(1)
                pixelwise_weight_sum += view_weight.unsqueeze(1)
        view_weights = torch.cat(view_weights_list, dim=1)  # [B, nview, H, W]

        #
        similarity = volume_sum.div_(pixelwise_weight_sum)
        del volume_sum

        # step 3. cost volume regularization
        prob_volume_pre = cost_regularization(similarity).squeeze(1)

        prob_volume = F.softmax(prob_volume_pre, dim=1)

        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]

        return {"depth": depth, "photometric_confidence": photometric_confidence, "prob_volume": prob_volume,
                "depth_values": depth_values, "view_weights": view_weights.detach()}


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    scaled_disp = min_disp + (max_disp - min_disp) * disp

    scaled_disp = scaled_disp.clamp(min=1e-4)
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    scaled_disp = 1 / depth

    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    disp = (scaled_disp - min_disp) / ((max_disp - min_disp))

    return disp


def upsample_depth(depth, mask, ratio=8):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)


class GetCost(nn.Module):
    def __init__(self):
        super(GetCost, self).__init__()

    def forward(self, depth_values, features, proj_matrices, depth_interval, depth_max, depth_min, CostNum=4,
                view_weights=[]):
        proj_matrices = torch.unbind(proj_matrices, 1)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        depth_values = 1. / depth_values
        _, feature_channel, height, width = ref_feature.size()

        if height != view_weights.size(2):
            view_weights = F.interpolate(view_weights,
                                         size=(height, width), mode='bilinear')

        if CostNum > 1:
            depth_range_samples = get_depth_range_samples(cur_depth=depth_values.squeeze(1),
                                                          ndepth=CostNum,
                                                          depth_inteval_pixel=depth_interval.squeeze(1),
                                                          dtype=ref_feature[0].dtype,
                                                          device=ref_feature[0].device,
                                                          shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                 ref_feature.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)
        else:
            depth_range_samples = depth_values
        depth_range_samples = 1. / depth_range_samples

        # step 2. differentiable homograph, build cost volume
        batch, feat_channel, height, width = features[0].shape
        dtype, device = features[0].dtype, features[0].device
        num_depth = depth_range_samples.shape[1]
        pixelwise_weight_sum = 1e-5 * torch.ones((batch, 1, 1, height, width), dtype=dtype, device=device)
        volume_sum = torch.zeros((batch, feat_channel, num_depth, height, width), dtype=dtype, device=device)
        idx = 0

        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_feature = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_range_samples)
            warped_volume = (warped_feature * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            view_weight = view_weights[:, idx].unsqueeze(1)
            idx += 1
            if self.training:
                volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
            else:
                volume_sum += warped_volume * view_weight.unsqueeze(1)
                pixelwise_weight_sum += view_weight.unsqueeze(1)

        similarity = volume_sum.div_(pixelwise_weight_sum)

        # aggregated matching cost across all the source views
        volume_aggregated = similarity.view(batch, feat_channel * num_depth, height, width)
        return volume_aggregated


class DI_MVS(nn.Module):
    def __init__(self, args, depth_interals_ratio=[4, 2, 1], stage_channel=True):
        super(DI_MVS, self).__init__()
        self.ndepths = args.ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.stage_channel = stage_channel
        seq_len = [int(e) for e in args.GRUiters.split(",")]
        self.seq_len = seq_len
        self.args = args
        self.num_stage = 3
        self.CostNum = args.CostNum
        self.GetCost = GetCost()
        self.hdim_stage = [64, 32, 16]
        self.cdim_stage = [64, 32, 16]
        self.context_feature = [128, 64, 32]
        self.feat_ratio = 2
        self.cost_dim_stage = [32, 16, 8]
        # self.G = 8
        self.cost_reg = args.cost_reg
        if self.cost_reg == 'small':
            self.cost_regularization = CostRegNet_small(in_channels=32, base_channels=8)
        elif self.cost_reg == 'regular':
            self.cost_regularization = CostRegNet(in_channels=32, base_channels=8)
        else:
            print("cost_reg type error!")

        print(
            "**********netphs:{}, depth_intervals_ratio:{}, hdim_stage:{}, cdim_stage:{}, context_feature:{}, cost_dim_stage:{}************".format(
                self.ndepths, depth_interals_ratio, self.hdim_stage, self.cdim_stage, self.context_feature,
                self.cost_dim_stage))
        self.feature = P_1to8_FeatureNet(base_channels=8, out_channel=self.cost_dim_stage,
                                         stage_channel=self.stage_channel)
        self.cnet_depth = P_1to8_FeatureNet(base_channels=8, out_channel=self.context_feature,
                                            stage_channel=self.stage_channel)
        # self.cnet_depth = MultiBasicEncoder()
        self.update_block_depth1 = BasicUpdateBlockDepth(hidden_dim=self.hdim_stage[0],
                                                         cost_dim=self.cost_dim_stage[0] * self.CostNum,
                                                         ratio=self.feat_ratio, context_dim=self.cdim_stage[0],
                                                         UpMask=True)
        self.update_block_depth2 = BasicUpdateBlockDepth(hidden_dim=self.hdim_stage[1],
                                                         cost_dim=self.cost_dim_stage[1] * self.CostNum,
                                                         ratio=self.feat_ratio, context_dim=self.cdim_stage[1],
                                                         UpMask=True)

        self.update_block_depth3 = BasicUpdateBlockDepth(hidden_dim=self.hdim_stage[2],
                                                         cost_dim=self.cost_dim_stage[2] * self.CostNum,
                                                         ratio=self.feat_ratio, context_dim=self.cdim_stage[2],
                                                         UpMask=True)
        self.update_block = nn.ModuleList(
            [self.update_block_depth1, self.update_block_depth2, self.update_block_depth3])
        self.depthnet = DepthNet()

        # self.depth_update_net = DepthUpdate(feat_channels=32, n_query_channels=16)
        self.pixelwise_net = PixelwiseNet(in_channels=1)

        self.confidence_head0 = nn.Sequential(
            nn.Conv2d(self.hdim_stage[0], 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=1),
        )
        self.confidence_head1 = nn.Sequential(
            nn.Conv2d(self.hdim_stage[1], 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=1),
        )
        self.confidence_head2 = nn.Sequential(
            nn.Conv2d(self.hdim_stage[2], 32, 3, stride=1, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, stride=1, padding=0, dilation=1),
        )

    def forward(self, imgs, proj_matrices, depth_values):
        disp_min = depth_values[:, 0, None, None, None]
        disp_max = depth_values[:, -1, None, None, None]
        depth_max_ = 1. / disp_min
        depth_min_ = 1. / disp_max

        self.scale_inv_depth = partial(disp_to_depth, min_depth=depth_min_, max_depth=depth_max_)

        depth_interval = (disp_max - disp_min) / depth_values.size(1)
        # step 1. feature extraction
        features = []
        depth_predictions = []
        photometric_confidences = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        cnet_depth = self.cnet_depth(imgs[:, 0])
        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            ref_feature = features_stage[0]
            if stage_idx == 0:
                depth_range_samples = get_depth_range_samples(cur_depth=depth_values,
                                                              ndepth=self.ndepths,
                                                              depth_inteval_pixel=self.depth_interals_ratio[
                                                                                      stage_idx] * depth_interval,
                                                              dtype=ref_feature[0].dtype,
                                                              device=ref_feature[0].device,
                                                              shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                     ref_feature.shape[3]],
                                                              max_depth=disp_max,
                                                              min_depth=disp_min)
                depth_range_samples = 1. / depth_range_samples

                init_depth = self.depthnet(features_stage, proj_matrices_stage, depth_values=depth_range_samples,
                                           num_depth=self.ndepths, cost_regularization=self.cost_regularization,
                                           pixelwise_net=self.pixelwise_net)

                view_weights = init_depth["view_weights"]
                photometric_confidence = init_depth["photometric_confidence"]
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),
                                                       [ref_feature.shape[2] * 8, ref_feature.shape[3] * 8],
                                                       mode='nearest')
                photometric_confidences.append(photometric_confidence.squeeze(1))
                prob_volume = init_depth['prob_volume']
                depth_values = init_depth['depth_values']
                init_depth = init_depth['depth']
                cur_depth = init_depth.unsqueeze(1)

                depth_predictions = [init_depth.squeeze(1)]
            else:
                cur_depth = depth_predictions[-1].unsqueeze(1)

            inv_cur_depth = depth_to_disp(cur_depth, depth_min_, depth_max_)

            cnet_depth_stage = cnet_depth["stage{}".format(stage_idx + 1)]

            hidden_d, inp_d = torch.split(cnet_depth_stage, [self.hdim_stage[stage_idx], self.cdim_stage[stage_idx]],
                                          dim=1)

            current_hidden_d = torch.tanh(hidden_d)

            inp_d = torch.relu(inp_d)

            depth_cost_func = partial(self.GetCost, features=features_stage, proj_matrices=proj_matrices_stage,
                                      depth_interval=depth_interval * self.depth_interals_ratio[stage_idx],
                                      depth_max=disp_max, depth_min=disp_min,
                                      CostNum=self.CostNum, view_weights=view_weights)

            current_hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block[stage_idx](current_hidden_d,
                                                                                          depth_cost_func,
                                                                                          inv_cur_depth,
                                                                                          inp_d, seq_len=self.seq_len[
                    stage_idx],
                                                                                          scale_inv_depth=self.scale_inv_depth,
                                                                                          )
            confidence = 0
            if stage_idx == 0:
                confidence = self.confidence_head0(current_hidden_d)
            elif stage_idx == 1:
                confidence = self.confidence_head1(current_hidden_d)
            elif stage_idx == 2:
                confidence = self.confidence_head2(current_hidden_d)
            else:
                print("TypeError")
            confidence = torch.sigmoid(confidence)
            photometric_confidences.append(confidence.squeeze(1))

            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                depth_predictions.append(self.scale_inv_depth(inv_depth_i)[1].squeeze(1))
            last_mask = up_mask_seqs[-1]
            last_inv_depth = inv_depth_seqs[-1]
            inv_depth_up = upsample_depth(last_inv_depth, last_mask, ratio=self.feat_ratio).unsqueeze(1)
            final_depth = self.scale_inv_depth(inv_depth_up)[1].squeeze(1)
            depth_predictions.append(final_depth)

        return {"depth": depth_predictions, "photometric_confidences": photometric_confidences,
                "prob_volume": prob_volume, "depth_values": depth_values}
