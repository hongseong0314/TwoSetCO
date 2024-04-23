import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, path=1, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        nS = model_params['I'] if path == 1 else model_params['J']
        nA = model_params['I'] if path == 0 else model_params['J']
        self.normalization_1 = nn.LayerNorm(nA)
        # self.normalization_1 = self.norm = nn.InstanceNorm3d(ms_hidden_dim, affine=True, track_running_stats=False)

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)
        Waa = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, ms_hidden_dim))
        self.Waa = nn.Parameter(Waa)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)

        self.edge = nn.Linear(head_num*ms_hidden_dim, 1)
        # shape: (head, 1)

        Wa = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((nS, nS))
        self.Wa = nn.Parameter(Wa)

    def forward(self, q, k_s, k_a, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k_a.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']
        sqrt_ms_dim = self.model_params['sqrt_ms_dim']

        dot_product_self = torch.matmul(q, k_s.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, row_cnt)

        dot_product_self = F.gelu(dot_product_self / sqrt_qkv_dim)
        # shape: (batch, head_num, row_cnt, row_cnt)
        dot_product_self_weight = nn.Softmax(dim=3)(dot_product_self)

        dot_product_another = torch.matmul(q, k_a.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_another = F.relu(dot_product_another / sqrt_qkv_dim)
        # shape: (batch, head_num, row_cnt, col_cnt)

        # dot_product_score = torch.matmul(dot_product_self.matmul(self.Wa)\
        #                                  , dot_product_another)
        dot_product_score = torch.matmul(dot_product_self_weight, dot_product_another)
        dot_product_score = dot_product_score / sqrt_qkv_dim
        
        cost_mat_score = cost_mat[:, None, :, :, None].expand(batch_size, head_num, row_cnt, col_cnt, 1)
        # shape: (batch, head_num, row_cnt, col_cnt)

        # two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        two_scores = torch.cat((dot_product_score[..., None], cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        depth_dot_product = torch.matmul(ms1_activated.matmul(self.Waa)\
                                         , ms1_activated.transpose(3, 4))
        depth_weights = nn.Softmax(dim=4)(depth_dot_product)
        ae_score = torch.matmul(depth_weights, ms1_activated)
        ae_score = ae_score + ms1_activated

        ae_score_pose = ae_score.transpose(3, 4)
        ae_score_pose_norm = self.normalization_1(ae_score_pose)
        ae_score = ae_score_pose_norm.transpose(4, 3)

        edge = ae_score.permute(0, 1, 3, 2, 4)
        edge_reshaped = edge.reshape(batch_size, row_cnt, col_cnt, -1)
        edge2 = self.edge(edge_reshaped).squeeze(3)
        
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat, edge2

class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1,2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class InstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input):
        # input.shape: (batch, problem, embedding)

        # shape: (batch, problem, embedding)

        transposed = input.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))