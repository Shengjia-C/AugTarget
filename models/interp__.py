import torch
import torch.nn as nn
import torch.nn.functional as F
#from thop import profile
import numpy as np
from tensorboardX import SummaryWriter

class global_Interp(nn.Module):

    def __init__(self, bs, W, H, inplanes, planes, stride=1):
        super(global_Interp, self).__init__()

        self.bs = bs
        self.C = inplanes
        self.W = W
        self.H = H
        self.sample_size = 32

        self.fc2 = nn.Linear(self.sample_size*self.sample_size, self.sample_size*self.sample_size).cuda()
        self.fc3 = nn.Linear(self.bs*self.sample_size*self.sample_size, self.sample_size*self.sample_size).cuda()
        self.relu = nn.ReLU(inplace=True)

        self.interp_pool = nn.AdaptiveMaxPool2d((self.W, self.H))
        self.upsample = nn.Upsample(size=[self.W, self.H], mode='bilinear', align_corners=True).cuda()

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(0.5)

        self.attconv = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(),
                nn.Conv2d(inplanes, 32, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.ReLU(),
            )

        num_parts = 9
        self.n_parts = num_parts
        self.grouping = GroupingUnit(inplanes, num_parts).cuda()
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        import pickle
#         self.ex_interp = pickle.load(open('/home/jxlab/workspace/csj/coco_graph_r.pkl', 'rb'))
#         self.ex_interp = self.ex_interp[1:, 1:]
#         self.ex_interp = torch.from_numpy(self.ex_interp).float().cuda()
        self.fc_in1 = nn.Linear(self.sample_size * self.sample_size, self.C).cuda()
        self.fc_in2 = nn.Linear(self.C, self.C).cuda()
        self.fc_ex1 = nn.Linear(self.C,self.sample_size * self.sample_size).cuda()
        self.fc_ex2 = nn.Linear(self.C,self.sample_size * self.sample_size).cuda()
        
        self.GAT = GAT(n_units=[16], n_heads=8, dropout=0.2, alpha=0.2)

        self.HeteGCNLayer = HeteGCNLayer(self.C,self.C)
        
        self.map = nn.Linear(self.sample_size * self.sample_size, self.sample_size * self.sample_size)
        
        
        self.writer = SummaryWriter(log_dir=self.save_folder)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #################################################



    def find_interp_part(self,x):
        bs = x.shape[0] 
        
        x2 = F.interpolate(x, size=[self.sample_size, self.sample_size])  # 8,256,32,32
        x3 = x2.view(bs * self.C, self.sample_size * self.sample_size)  # 2048,1024   1280,1024
        x3 = self.fc2(x3)  # 2048,1024
        
        in_interp = self.fc_in1(x3) #2048, 256   2048,256
        
        in_interp = in_interp.view(bs,self.C,self.C)
        
        in_interp = in_interp[0] #256,256
        
        in_interp = self.fc_in2(in_interp.transpose(0, -1)) #128,128
        
        
        
        ex_interp = self.HeteGCNLayer(in_interp, in_interp) #256,256
        
        
        self.writer.add_image('Image', ex_interp)##################################################
        
        ex_interp = self.fc_ex1(ex_interp)
        ex_interp = self.fc_ex2(ex_interp.transpose(0,-1)) #1024,1024
        #x3 = x3 * self.ex_interp
        x3 = self.map(ex_interp) 
        
        templates = x3.view(self.sample_size * self.sample_size, self.sample_size, self.sample_size)
        indices = F.max_pool2d(x2, self.sample_size, return_indices=True)[1].squeeze() #2,48
        selected_templates = torch.stack([templates[i] for i in indices], 0) #2,48,14,14

        interp = ex_interp.unsqueeze(0).unsqueeze(0)
        interp = F.interpolate(interp, size=[self.sample_size, self.sample_size])
        interp = interp.repeat(bs,self.C,1,1)
        
        selected_templates = selected_templates.view(bs,self.C,self.sample_size,self.sample_size)
        selected_templates = selected_templates[0]
        selected_templates = selected_templates.repeat(bs,1,1,1)
        
        selected_templates = selected_templates * interp
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        selected_templates = self.GAT(selected_templates)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        x2 = F.relu(x2 * selected_templates)  #5,128,32,32.   8,128,32,32
        x2 = self.upsample(x2) #2,48,W,H

        interp = F.max_pool2d(x2, 1, 1)
        interp[interp < self.fuse_weight_1] = 0
        interp[interp >= self.fuse_weight_1] = 1
        interp = interp * 1 / (interp.sum(0) + 1e-6)
        interp = x * interp

        return interp


    def forward(self, x):

        feature = x
        global_interp = self.find_interp_part(feature)
        global_interp = self.interp_pool(global_interp)  # 1,96,W,H
        global_interp = x + global_interp

        region_feature, assign = self.grouping(x)
        region_feature = region_feature.contiguous().unsqueeze(3)  # 2,96,9,1

        att_list = []
        att = self.attconv(region_feature)
        att = F.softmax(att, dim=2)
        att_list.append(att)

        att_interp = region_feature * att_list[0]  # 2, 96, 9, 1
        att_interp = att_interp.contiguous().squeeze(3)  # 2,96,9

        # average all region features into one vector based on the attention
        att_interp = F.avg_pool1d(att_interp, self.n_parts) * self.n_parts
        att_interp = att_interp.contiguous().unsqueeze(3)  # 2,96,1,1
        att_interp = att_interp.expand(-1, -1, self.W, self.H)

        global_interp = global_interp * att_interp
        out = x + global_interp

        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class HeteGCNLayer(nn.Module):

    def __init__(self, in_layer_shape, out_layer_shape, type_fusion='att', type_att_size=64):
        super(HeteGCNLayer, self).__init__()


        self.in_layer_shape = in_layer_shape
        self.out_layer_shape = out_layer_shape

        self.hete_agg = nn.ModuleDict()

        self.hete_agg = HeteAggregateLayer(in_layer_shape, out_layer_shape, type_fusion,
                                                  type_att_size)

    def forward(self, x_dict, adj_dict):

        ret_x_dict = self.hete_agg(x_dict, adj_dict)

        return ret_x_dict


class HeteAggregateLayer(nn.Module):

    def __init__(self, in_layer_shape, out_shape, type_fusion, type_att_size):
        super(HeteAggregateLayer, self).__init__()


        self.type_fusion = type_fusion

       # self.W_rel = nn.ParameterDict()

        self.W_rel = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
        nn.init.xavier_uniform_(self.W_rel.data, gain=1.414)

        self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

        self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        if type_fusion == 'att':
            self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
            self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
            self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

    def forward(self, x_dict, adj_dict):

        self_ft = torch.mm(x_dict, self.w_self)

        nb_ft_list = [self_ft]


        nb_ft = torch.mm(x_dict, self.W_rel)
        #print(adj_dict.shape, nb_ft.shape)
        nb_ft = torch.spmm(adj_dict, nb_ft)
        nb_ft_list.append(nb_ft)


        if self.type_fusion == 'mean':
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)

        elif self.type_fusion == 'att':
            att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
            att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
            att_input = torch.cat([att_keys, att_query], 1)
            att_input = F.dropout(att_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(att_input, self.w_att))
            attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0, 1), dim=1)
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)


        output = agg_nb_ft + self.bias

        return output


import math
import numpy as np



class GroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(batch_size,
                                                                                                     self.num_parts,
                                                                                                     self.in_channels)

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1) #1,9,152,100
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3) #~~~~~~~~~~~~~~~~~~~~W,H->1,1
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)
        outputs_t = outputs.permute(0, 2, 1)

        # generate assignment map for basis for visualization
        return outputs_t, assign

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'




def featuremap_visual(feature,
                      out_dir='./vis/',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                      nrow=8,  # 每行显示多少个特征图通道
                      padding=10,  # 特征图之间间隔多少像素值
                      pad_value=1  # 特征图之间的间隔像素
                      ):
    import matplotlib.pylab as plt
    import torchvision
    import os
    import cv2
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)

    if c > num_ch > 0:
        feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))
    images = cv2.resize(images, (200, 200))
    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('global-hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

    plt.title(title)
    plt.imshow(images)
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
        plt.savefig(out_root, dpi=400)

    if show_feature:        plt.show()



############################################################

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn



class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):

        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.relu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x

############################################################



# def main():
#     x = torch.randn(8, 128, 128, 128).cuda() #8, 3, 256, 256
#     bs = x.shape[0]
#     C = x.shape[1]
#     W = x.shape[2]
#     H = x.shape[3]

#     Interpretation = global_Interp(bs, W, H, C, C).cuda()
#     y = Interpretation(x)
#     

#     # flops, params = profile(Interpretation, inputs=(x, ))
#     

# if __name__ == "__main__":
#     main()

