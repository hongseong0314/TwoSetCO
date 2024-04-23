import torch
import torch.nn as nn
import torch.nn.functional as F

from model.submodel import *

class SchFFSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        stage_cnt = self.model_params['stage_cnt']
        self.stage_models = nn.ModuleList([OneStageModel(stage_idx, **model_params) for stage_idx in range(stage_cnt)])

    def pre_forward(self, reset_state):
        stage_cnt = self.model_params['stage_cnt']
        for stage_idx in range(stage_cnt):
            problems = reset_state.problems_list[stage_idx]
            model = self.stage_models[stage_idx]
            model.pre_forward(problems)

    def soft_reset(self):
        # Nothing to reset
        pass

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        stage_cnt = self.model_params['stage_cnt']
        action_stack = torch.empty(size=(batch_size, pomo_size, stage_cnt), dtype=torch.long)
        prob_stack = torch.empty(size=(batch_size, pomo_size, stage_cnt))

        for stage_idx in range(stage_cnt):
            model = self.stage_models[stage_idx]
            action, prob = model(state)

            action_stack[:, :, stage_idx] = action
            prob_stack[:, :, stage_idx] = prob

        gathering_index = state.stage_idx[:, :, None]
        # shape: (batch, pomo, 1)
        action = action_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        prob = prob_stack.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        return action, prob

class OneStageModel(nn.Module):
    def __init__(self, stage_idx, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = FFSP_Encoder(**model_params)
        self.decoder = FFSP_Decoder(**model_params)

        self.encoded_col = None
        # shape: (batch, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch, job_cnt, embedding)
        self.position_embedding = PositionalEncoding(**model_params)
    def pre_forward(self, problems):
        # problems.shape: (batch, job_cnt, machine_cnt)
        batch_size = problems.size(0)
        job_cnt = problems.size(1)
        machine_cnt = problems.size(2)
        embedding_dim = self.model_params['embedding_dim']

        row_emb = torch.zeros(size=(batch_size, job_cnt, embedding_dim))
        # shape: (batch, job_cnt, embedding)
        col_emb = torch.zeros(size=(batch_size, machine_cnt, embedding_dim))
        # shape: (batch, machine_cnt, embedding)
        X_emb = torch.zeros(size=(batch_size, job_cnt, machine_cnt))

        seed_cnt = self.model_params['one_hot_seed_cnt']
        rand = torch.rand(batch_size, seed_cnt)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :machine_cnt]

        b_idx = torch.arange(batch_size)[:, None].expand(batch_size, machine_cnt)
        m_idx = torch.arange(machine_cnt)[None, :].expand(batch_size, machine_cnt)
        col_emb[b_idx, m_idx, rand_idx] = 1
        # col_emb = self.position_embedding(col_emb)
        # shape: (batch, machine_cnt, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems, X_emb)
        # encoded_row.shape: (batch, job_cnt, embedding)
        # encoded_col.shape: (batch, machine_cnt, embedding)

        self.decoder.set_kv(self.encoded_row)

    def forward(self, state):

        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        encoded_current_machine = self._get_encoding(self.encoded_col, state.stage_machine_idx)
        # shape: (batch, pomo, embedding)
        all_job_probs = self.decoder(encoded_current_machine,
                                     ninf_mask=state.job_ninf_mask)
        # shape: (batch, pomo, job)

        if self.training or self.model_params['eval_type'] == 'softmax':
            dist = torch.distributions.categorical.Categorical(probs=all_job_probs)
            job_selected = dist.sample()
            job_log_p = dist.log_prob(job_selected)
        else:
            job_selected = all_job_probs.argmax(dim=2)
            # shape: (batch, pomo)
            job_log_p = torch.zeros(size=(batch_size, pomo_size))  # any number is okay

        return job_selected, job_log_p

    def _get_encoding(self, encoded_nodes, node_index_to_pick):
        # encoded_nodes.shape: (batch, problem, embedding)
        # node_index_to_pick.shape: (batch, pomo)

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = self.model_params['embedding_dim']

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        # shape: (batch, pomo, embedding)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo, embedding)

        return picked_nodes
    
class FFSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        # self.X_embedding_dim = nn.Linear(4, 4)
    def forward(self, row_emb, col_emb, cost_mat, X_emb):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        # cost_mat = F.relu(self.X_embedding_dim(cost_mat))
        for layer in self.layers:
            # X = torch.stack((cost_mat, X_emb), dim=3)
            row_emb, col_emb, cost_mat = layer(row_emb, col_emb, cost_mat)
            # cost_mat = F.relu(self.X_embedding_dim(cost_mat + layer_cost_mat))
        return row_emb, col_emb

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(path=1,**model_params)
        self.col_encoding_block = EncodingBlock(path=0,**model_params)
        # Wr = torch.torch.distributions.Uniform(low=-0.5, high=0.5).sample((1,1))
        # Wc = torch.torch.distributions.Uniform(low=-0.5, high=0.5).sample((1,1))
        # self.Wr = nn.Parameter(Wr)
        # self.Wc = nn.Parameter(Wc)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out, row_edge_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out, col_edge_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))
        # edge_out = (row_edge_out + col_edge_out.transpose(1, 2)) / 2
        # edge_out = self.Wr * row_edge_out + self.Wc * col_edge_out.transpose(1, 2)
        return row_emb_out, col_emb_out, cost_mat

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
        self.SA_MHA = SAMixedScore_MultiHeadAttention(path=path,**model_params)
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

        out_concat, edge_out = self.SA_MHA(q, k_s, k_a, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        # out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        # out1 = row_emb + multi_head_out
        # out2 = self.normalization_2(out1)
        # out3 = self.feed_forward(out2)
        # out4 = out3 + out1
        out1 = self.normalization_1(row_emb + multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.normalization_2(out1 + out2)
        return out3, edge_out
        # shape: (batch, row_cnt, embedding)

class PreEncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        I = model_params['I']
        J = model_params['J']
        self.row_encoding_block = PreEncodingBlock(**model_params)
        self.col_encoding_block = PreEncodingBlock(**model_params)
        Wr = torch.torch.distributions.Uniform(low=-0.5, high=0.5).sample((1,1))
        Wc = torch.torch.distributions.Uniform(low=-0.5, high=0.5).sample((1,1))
        self.Wr = nn.Parameter(Wr)
        self.Wc = nn.Parameter(Wc)

        self.rnormalization_1 = nn.LayerNorm(embedding_dim)#InstanceNormalization(**model_params)
        self.rfeed_forward = FeedForward(**model_params)
        self.rnormalization_2 = nn.LayerNorm(embedding_dim)#InstanceNormalization(**model_params)

        self.cnormalization_1 = nn.LayerNorm(embedding_dim)#InstanceNormalization(**model_params)
        self.cfeed_forward = FeedForward(**model_params)
        self.cnormalization_2 = nn.LayerNorm(embedding_dim)#InstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_norm = self.rnormalization_1(row_emb)
        col_emb_norm = self.cnormalization_1(col_emb)

        row_emb_out, row_edge_out = self.row_encoding_block(row_emb_norm, col_emb_norm, cost_mat)
        col_emb_out, col_edge_out = self.col_encoding_block(col_emb_norm, row_emb_norm, cost_mat.transpose(1, 2))
        
        # row update
        rout1 = row_emb + row_emb_out
        rout2 = self.rnormalization_2(rout1)
        rout3 = self.rfeed_forward(rout2)
        rout4 = rout3 + rout1

        # col update
        cout1 = col_emb + col_emb_out
        cout2 = self.cnormalization_2(cout1)
        cout3 = self.cfeed_forward(cout2)
        cout4 = cout3 + cout1
        
        # edge_out = (row_edge_out + col_edge_out.transpose(1, 2)) / 2
        edge_out = self.Wr * row_edge_out + self.Wc * col_edge_out.transpose(1, 2)
        return rout4, cout4, edge_out

class PreEncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)


    def forward(self, row_emb, col_emb, cost_mat):
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat, edge_out = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        return multi_head_out, edge_out
        # shape: (batch, row_cnt, embedding)

class FFSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        # I = model_params['I']
        # J = model_params['J']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        # self.rnormalization_1 = nn.LayerNorm(embedding_dim)
        self.cnormalization_1 = InstanceNormalization(**model_params)
        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        batch_size = encoded_jobs.size(0)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        encoded_no_job = self.encoded_NO_JOB.expand(size=(batch_size, 1, embedding_dim))
        encoded_jobs_plus_1 = torch.cat((encoded_jobs, encoded_no_job), dim=1)
        # shape: (batch, job_cnt+1, embedding)
        # encoded_jobs_plus_1 = self.rnormalization_1(encoded_jobs_plus)
        self.k = reshape_by_heads(self.Wk(encoded_jobs_plus_1), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs_plus_1), head_num=head_num)
        # shape: (batch, head_num, job+1, qkv_dim)
        self.single_head_key = encoded_jobs_plus_1.transpose(1, 2)
        # shape: (batch, embedding, job+1)

    def forward(self, encoded_machine, ninf_mask):
        # encoded_machine.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job_cnt+1)

        head_num = self.model_params['head_num']
        # encoded_machine_norm = self.cnormalization_1(encoded_machine)
        #  Multi-Head Attention
        #######################################################
        encoded_machine = self.cnormalization_1(encoded_machine)
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v,
                                                            rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)
        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, job_cnt+1)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job_cnt+1)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, job_cnt+1)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, qkv_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch, head_num, job_cnt+1, qkv_dim)
        # rank2_ninf_mask.shape: (batch, job_cnt+1)
        # rank3_ninf_mask.shape: (batch, n, job_cnt+1)

        batch_size = q.size(0)
        n = q.size(2)
        job_cnt_plus_1 = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, job_cnt+1)

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, job_cnt_plus_1)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, job_cnt_plus_1)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, job_cnt+1)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)
        # shape: (batch, n, head_num*qkv_dim)

        return out_concat


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

class PositionalEncoding(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.d_model = model_params['embedding_dim']
        # self.device = device
        self.machine_num = model_params['J'] #machine_num

        self.encoding = torch.zeros(self.machine_num, self.d_model, requires_grad=False)

        pos = torch.arange(0, self.machine_num)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, self.d_model, step=2).float()

        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / self.d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.d_model)))

    def forward(self, machine_embedding):
        batch_size = machine_embedding.size(0)
        return machine_embedding + \
            self.encoding[None, ...].expand(batch_size, self.machine_num, self.d_model)