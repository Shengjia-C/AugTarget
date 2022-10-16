import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
#from thop import profile
import numpy as np
#from tensorboardX import SummaryWriter
from collections import OrderedDict

class global_Interp(nn.Module):

    def __init__(self, bs, inplanes,  W, H,  stride=1):
        super(global_Interp, self).__init__()

        self.bs = bs
        self.C = inplanes
        self.W = W
        self.H = H
        self.sample_size = 64
        self.code_len = 128
        
        self.fc_encode1 = nn.Linear(self.sample_size*self.sample_size, self.code_len).cuda()
        self.fc_encode2 = nn.Linear(self.bs*self.C, self.code_len).cuda()
        self.fc_encode3 = nn.Linear(4096, self.code_len).cuda()
        
        self.fc3 = nn.Linear(self.bs*self.sample_size*self.sample_size, self.sample_size*self.sample_size).cuda()
        

        self.pool1 = nn.AdaptiveMaxPool2d((self.W, self.H))
        self.pool2 = nn.AdaptiveMaxPool2d((self.W, self.H))
        
        self.fc_decode1 = nn.Linear(self.code_len,self.W * self.W).cuda()
        self.fc_decode2 = nn.Linear(self.code_len,self.bs * self.C).cuda()
        

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
       
        import pickle
#         self.ex_interp = pickle.load(open('/home/jxlab/workspace/csj/coco_graph_r.pkl', 'rb'))
#         self.ex_interp = self.ex_interp[1:, 1:]
#         self.ex_interp = torch.from_numpy(self.ex_interp).float().cuda()
 
        self.GNN = GraphNetwork(self.code_len,self.code_len).cuda()
        
        self.HeteGCNLayer1 = HeteGCNLayer(self.code_len,self.code_len).cuda()
        self.HeteGCNLayer2 = HeteGCNLayer(self.code_len,self.code_len).cuda()
        
        self.fc_a1 = nn.Linear(self.C,1).cuda()
        self.fc_a2 = nn.Linear(self.C, self.bs * self.C).cuda()
        
        #self.writer = SummaryWriter(log_dir='/home/jxlab/workspace/csj/可视化')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #################################################
    


    def forward(self, x):
        
        x2 = F.interpolate(x, size=[self.sample_size, self.sample_size])  # 8,256,64,64
        x2 = x2.view(-1, self.sample_size * self.sample_size)  # 2048,4096 
        x2 = self.fc_encode1(x2).transpose(0,-1)  # 128,2048
        x2 = self.fc_encode2(x2) #128,128
        
        in_interp = torch.tanh(x2)
        in_interp = torch.sign(in_interp).unsqueeze(0)
        # _, index = x2.sort(0, descending=True)
        # N, D = x2.shape
        # B_creat = torch.cat((torch.ones([int(N/2), D]), -torch.ones([N - int(N/2), D]))).cuda()    
        # in_interp = torch.zeros(x2.shape).cuda().scatter_(0, index, B_creat)
        
       
        
        #########
        global_semantic = self.pool1(x)  # 8, 256, 64, 64
        att_list = []
        att = self.attconv(global_semantic)
        att = F.softmax(att, dim=2)
        att_list.append(att)
        att_interp = global_semantic * att_list[0]  
        att_interp = att_interp.view(self.bs*self.C,self.W,self.H)  # 2048, 64, 64
        
        # average all region features into one vector based on the attention
        att_interp = F.avg_pool1d(att_interp, self.W//2) * self.W//2
        att_interp = att_interp.view(self.code_len,-1) #128,2048  #128, 4096
        att_interp = self.fc_encode3(att_interp) #128,128
        global_interp = att_interp * x2
        global_interp = torch.tanh(global_interp)
        global_interp = torch.sign(global_interp).unsqueeze(0)
        #########
        
        edge_feat, node_feat = self.GNN(in_interp, global_interp)
        edge_feat = edge_feat.squeeze()
        node_feat = node_feat.squeeze()
        node_feat = self.HeteGCNLayer1(node_feat,edge_feat) #128,128
        interp = self.HeteGCNLayer2(node_feat,edge_feat) #128,128
        interp = self.fc_decode1(interp).transpose(0,-1) #1024,128
        interp = self.fc_decode2(interp) #1024,4096
        
        interp = interp.view(-1, self.C) #8192, 512
        coefficient = self.fc_a1(interp) #8192,1
        
        ccoefficient = torch.transpose(torch.transpose(
            coefficient.view(self.bs, self.W, self.H),0,2),1,1).reshape(self.bs, self.W, -1)#8, 32, 32
        coefficient = coefficient.view(self.bs, 1, self.W, self.H).repeat(1,self.C,1,1)
        
        interp = x * coefficient
        interp = torch.sum(interp,0) #512, 32, 32
        interp = torch.tanh(interp)
        interp = torch.sign(interp)
        
        interp = interp.view(-1,self.W*self.H).transpose(0,-1) #1024,512
        interp = self.fc_a2(interp) #512,8192
        interp = torch.relu(interp)
        interp = interp.view(self.bs,self.C,self.W,self.H)
        
        interp = torch.tanh(interp + x)
        interp =  interp * x
        x = interp + x

        return x


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



class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout
        
        ##########################
        self.input_dim = 256
        
        input_dim = self.input_dim
        self.fc_eq3_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq3_u = nn.Linear(input_dim, input_dim)
        self.fc_eq4_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq4_u = nn.Linear(input_dim, input_dim)
        self.fc_eq5_w = nn.Linear(2*input_dim, input_dim)
        self.fc_eq5_u = nn.Linear(input_dim, input_dim)
        
        ##########################
        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 2,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)
        
    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        #print('num_tasks',num_tasks,'num_data',num_data) #1,466
        
        # get eye matrix (batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).cuda() #1,466,466
        #print('diag_mask',diag_mask.shape) #1, 466, 466
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1) 
        #print('edge_feat',edge_feat.shape) #1, 466, 466
        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)
        #print('aggr_feat',aggr_feat.size()) #1, 466, 300
        #print('node_feat1',node_feat.shape) #1, 466, 300
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        # node_feat = ((0*node_feat + 2*aggr_feat)/2).transpose(1,2)
        
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        
        
        # for m in self.network.children():
        #     return m(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        
       
        return node_feat
    

class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            # layers
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                # set layer
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # compute abs(x_i, x_j)
        num_tasks = node_feat.size(0) 
        num_data = node_feat.size(1)
        
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        
        x_ij = torch.transpose(x_ij, 1, 3)
#        x_ij = torch.transpose(x_ij, 1, 2)#################

        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1)

        
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).cuda()################
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)

        return edge_feat

class GraphNetwork(nn.Module):
    def __init__(self, node_features, edge_features):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = node_features #######################
        self.node_features = node_features########################
        self.edge_features = edge_features
        self.num_layers = 2
        self.dropout = 0.2

        # for each layer
        for l in range(self.num_layers):
            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            # set edge to node
            edge2node_net = NodeUpdateNetwork(
                in_features=self.in_features if l == 0 else self.node_features,
                num_features=self.node_features ,#!!!!!!!!!!!!!!!!!calss
                dropout=self.dropout if l < self.num_layers - 1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, init_node_feat, init_edge_feat): #####################
        # for each layer
        edge_feat_list = []
        node_feat_list = []
        node_feat = init_node_feat
        edge_feat = init_edge_feat
        

        for l in range(self.num_layers):
            # (1) edge update
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)
            
            
            
            
            # save edge feature
#            edge_feat_list.append(edge_feat)
#            node_feat_list.append(node_feat)

        return node_feat,edge_feat





# def featuremap_visual(feature,
#                       out_dir='./vis/',  # 特征图保存路径文件
#                       save_feature=True,  # 是否以图片形式保存特征图
#                       show_feature=True,  # 是否使用plt显示特征图
#                       feature_title=None,  # 特征图名字，默认以shape作为title
#                       num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
#                       nrow=8,  # 每行显示多少个特征图通道
#                       padding=10,  # 特征图之间间隔多少像素值
#                       pad_value=1  # 特征图之间的间隔像素
#                       ):
#     import matplotlib.pylab as plt
#     import torchvision
#     import os
#     import cv2
#     # feature = feature.detach().cpu()
#     b, c, h, w = feature.shape
#     feature = feature[0]
#     feature = feature.unsqueeze(1)

#     if c > num_ch > 0:
#         feature = feature[:num_ch]

#     img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
#     img = img.detach().cpu()
#     img = img.numpy()
#     images = img.transpose((1, 2, 0))
#     images = cv2.resize(images, (200, 200))
#     # title = str(images.shape) if feature_title is None else str(feature_title)
#     title = str('global-hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

#     plt.title(title)
#     plt.imshow(images)
#     if save_feature:
#         # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
#         # plt.savefig(os.path.join(root,'1.jpg'))
#         out_root = title + '.jpg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.jpg')
#         plt.savefig(out_root, dpi=400)

#     if show_feature:        plt.show()




