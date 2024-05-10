import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from model.submodel import *

class InferenceSchedule(object):
  def __init__(self, **diffusion_params):
    self.inference_schedule = diffusion_params['inference_schedule']
    self.T = diffusion_params['T']
    self.inference_T = diffusion_params['inference_T']

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))

class CategoricalDiffusion(object):
  def __init__(self, **diffusion_params):
    # Diffusion steps
    self.T = diffusion_params['T']
    # Noise schedule
    if diffusion_params['schedule'] == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, self.T)
    elif diffusion_params['schedule'] == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, self.T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))

    matrix_prob = xt[..., 1].clamp(0, 1)
    row_weight = matrix_prob.softmax(dim=1)
    xt1 = torch.bernoulli(matrix_prob)
    mask = xt1 * F.one_hot((xt1 * row_weight).max(dim=2)[1], num_classes=xt.size(1)).to(xt1.device)
    xt1 = (xt1 * mask).detach()
    return matrix_prob * mask, xt1.long()
  
class DSAATSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = SAEncoder(**model_params)
        self.decoder = ATSP_Decoder(**model_params)

        self.encoded_row = None
        self.encoded_col = None

        # shape: (batch, node, embedding)
        self.diffusion = CategoricalDiffusion(**model_params['diffusion'])
        self.time_schedule = InferenceSchedule(**model_params['diffusion'])

    def pre_forward(self, reset_state):

        problems = reset_state.problems
        # problems.shape: (batch, node, node)

        batch_size = problems.size(0)
        node_cnt = problems.size(1)
        embedding_dim = self.model_params['embedding_dim']

        row_emb = torch.zeros(size=(batch_size, node_cnt, embedding_dim))
        # emb.shape: (batch, node, embedding)
        col_emb = torch.zeros(size=(batch_size, node_cnt, embedding_dim))
        # shape: (batch, node, embedding)

        seed_cnt = self.model_params['one_hot_seed_cnt']
        rand = torch.rand(batch_size, seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :node_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, node_cnt)
        n_idx = torch.arange(node_cnt)[None, :].expand(batch_size, node_cnt)
        col_emb[b_idx, n_idx, rand_idx] = 1
        # shape: (batch, node, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems)
        # encoded_nodes.shape: (batch, node, embedding)

        self.decoder.set_kv(self.encoded_col)

    def forward(self, t, xt):
        # xt = 1#state.xt
        t1, t2 = self.time_schedule(t)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        t1 = torch.from_numpy(t1).view(1)
        t2 = torch.from_numpy(t2).view(1)

        x0_pred = self.decoder(t1.float(), xt)
        matrix_prob, xt = self.posterior(t1, t2, xt, x0_pred)
        return matrix_prob, xt
    
    def posterior(self, t1, t2, xt, x0_pred):
        diffusion = self.diffusion
        Q_t = np.linalg.inv(diffusion.Q_bar[t2]) @ diffusion.Q_bar[t1]
        Q_t = torch.from_numpy(Q_t).float().to(x0_pred.device)
        # # else:
        # #   Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t1]).float().to(x0_pred.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[t2]).float().to(x0_pred.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred[..., 1]

        # if target_t > 0:
        #     xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        # else:
        #     xt = sum_x_t_target_prob.clamp(min=0)
        
        sum_x_t_target_prob = sum_x_t_target_prob.clamp(0, 1)
        row_weight = sum_x_t_target_prob.softmax(dim=1)
        xt1 = torch.bernoulli(sum_x_t_target_prob)
        mask = xt1 * F.one_hot((xt1 * row_weight).max(dim=2)[1], num_classes=xt1.size(1)).to(xt1.device)
        xt1 = (xt1 * mask).detach()
        return sum_x_t_target_prob, xt1.long() #* mask

########################################
# ENCODER
########################################
class SAEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out, _ = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out, _ = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out

class EncodingBlock(nn.Module):
    def __init__(self, path=1, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Ws = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wa = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = SAMixedScore_MultiHeadAttention(path=path,**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.normalization_1 = InstanceNormalization(**model_params)#nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(**model_params)
        self.normalization_2 = InstanceNormalization(**model_params)#nn.LayerNorm(embedding_dim)

    def forward(self, row_emb, col_emb, cost_mat):
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k_s = reshape_by_heads(self.Ws(row_emb), head_num=head_num)
        k_a = reshape_by_heads(self.Wa(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat, edge_out = self.mixed_score_MHA(q, k_s, k_a, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.normalization_1(row_emb + multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.normalization_2(out1 + out2)
        return out3, edge_out
        # shape: (batch, row_cnt, embedding)

########################################
# Decoder
########################################

class ATSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        time_embed_dim = embedding_dim // 2
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        # self.single_head_key = None  # saved key, for single-head attention
        self.q = None  # saved q1, for multi-head attention

        self.node_to_node = nn.Sequential(
        # nn.ReLU(),
        nn.Conv2d(2, 2, kernel_size=1, bias=True))

        self.time_embed = nn.Sequential(
        nn.Linear(embedding_dim, time_embed_dim),
        nn.ReLU(),
        nn.Linear(time_embed_dim, embedding_dim),
        )
    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        head_num = self.model_params['head_num']

        self.q = reshape_by_heads(self.Wq(encoded_jobs), head_num=head_num)
        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=head_num)
        # shape: (batch, head_num, job, qkv_dim)

        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job)

    def forward(self, t, xt):
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)

        head_num = self.model_params['head_num']
        t_emb = self.time_embed(timestep_embedding(t, \
                                                   self.model_params['embedding_dim']))
        # shape: (batch, embedding)
        #  Multi-Head Attention
        #######################################################
        out_concat = self._multi_head_attention(self.q, self.k, self.v)
        # shape: (batch, job, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        mh_atten_out = mh_atten_out + t_emb[:, None, :]
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # score_masked = score_clipped + ninf_mask

        # probs = F.softmax(score_masked, dim=2)
        # # shape: (batch, node, node)
        score = torch.stack((score_clipped, xt), dim=1)
        score_probs = self.node_to_node(score)
        score_probs = score_probs.permute(0, 2, 3, 1)
        return score_probs.softmax(dim=3)

    def _multi_head_attention(self, q, k, v):
        # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or pomo
        # k,v shape: (batch, head_num, node, key_dim)
        # rank2_ninf_mask.shape: (batch, node)
        # rank3_ninf_mask.shape: (batch, group, node)

        batch_s = q.size(0)
        n = q.size(2)
        node_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, node)

        score_scaled = score / sqrt_qkv_dim
        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, node)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, key_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, key_dim)

        out_concat = out_transposed.reshape(batch_s, n, head_num * qkv_dim)
        # shape: (batch, n, head_num*key_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def get_rewards(solutions, problems, num_nodes=20):
    tours = solutions.argmax(dim=2)
    solution_len = (problems * solutions).sum(dim=2).sum(dim=1)
    for idx, tour in enumerate(tours):
        if (np.sort(tour) == np.arange(num_nodes)).all():
            pass#solution_len[idx] -= 1
        else:
            solution_len[idx] = 10
    return -solution_len

def get_x_T(batch_size, num_nodes):
    result = torch.empty(0, num_nodes, dtype=torch.long)
    for _ in range(batch_size):
        row = torch.randperm(num_nodes)
        row = row.view(1, -1)
        result = torch.cat((result, row), dim=0)
    xT = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.float32)
    for idx, tour in enumerate(result):
        for i in range(tour.shape[0]):
            xT[idx, tour[i%num_nodes], tour[(i + 1)%num_nodes]] = 1
    return xT

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding