"""
the shape of attention block

    input: x -> shape (1, embedding_dim)

ps: for linear algebra understanding, I will use (Wq @ x.T) instead of (x @ Wq)
? why not (Wq @ x) ? because mathematically, the result is a matrix with parallel result between each row (Rank = 1), the express ability is much weaker than a column vector with Each row of Wq defines a learned linear feature detector. Each component of q is its response.

the actual purpose of (Wq @ x.T), is to do a linear transform to the original x space into a  query space, with each dim meaning one query

    W_query : Wq    -> shape (query_dim, embedding_dim)
    W_key : Wk      -> shape (key_dim, embedding_dim)

    query_dim == key_dim == value_dim
    W_value : Wv    -> shape (value_dim, embedding_dim)

for parallel computing:

    #           ┌───────────────────────────────┐
    #           │  ───────── Wq_0 ───────────   │
    #           │  ───────── Wq_1 ───────────   │
    # Wq    =   │  ───────── Wq_2 ───────────   │
    #           └───────────────────────────────┘
    #
    #           ┌    │         │         │      ┐
    # X     =   │   x_0       x_1       x_2     │
    #           └    │         │         │      ┘
    #
    #           ┌───────────────────────────────┐
    #           │  ─────── Wq_0 @ X ────────    │
    # Q     =   │  ─────── Wq_1 @ X ────────    │
    #           │  ─────── Wq_2 @ X ────────    │
    #           └───────────────────────────────┘
    #

    X   -> (embedding_dim, seq_length) , this is actually transposed from (1, embedding_dim) then combined, each column is xi
    Wq  -> (query_dim, embedding_dim)
    Q   = Wq @ X -> (query_dim, seq_length) , row: one query for many x; column: one x for queries

    Wk  -> (key_dim, embedding_dim)
    K   = Wk @ X -> (key_dim, context_size)

    query_dim == key_dim
    Attention Scores = Q @ K.T / sqrt(qkv_dim) -> (context_size, context_size) - each row: query, each column: key

    why sqrt(d)? - sqrt(d) represents the standard deviation. For the softmax function, if the structure of the input numbers remains unchanged but the multiplier changes, the distribution of the output will change; this is positively correlated with sqrt(d). Modifying this divisor can be viewed as adjusting the temperature.

Single Head:
    query_dim == key_dim == value_dim
    Wv -> (value_dim, embedding_dim)
    V  = Wv @ X -> (value_dim, context_size)

    A  = softmax(Scores, ...) -> (context_size, context_size)
    output = V @ A.T -> (value_dim, context_size)

Batch (with nn.Linear, which will do math in last dimension):
    nn.Linear -> x @ W.T + bias

    x  -> (batch_size, seq_length, embedding_dim)
    wq -> (query_dim, embedding_dim)
    Wk -> (key_dim, embedding_dim)
    Wv -> (value_dim, embedding_dim)

    # nn.Linear applies the same transformation
    # independently to every Token in every Batch.

    Q = W_query(x) -> (batch_size, seq_length, query_dim)
    K = W_key(x)   -> (batch_size, seq_length, key_dim)
    V = W_value(x) -> (batch_size, seq_length, value_dim)

    Attention Scores:
    K_transposed = K.transpose(-2, -1)
        -> (batch_size, key_dim, seq_length)

    Scores = Q @ K_transposed / sqrt(qkv_dim)
        -> (batch_size, seq_length, seq_length) - row : query, column : key

    Causal Mask:
    Mask = triu(ones(context_size, context_size, dtype=torch.bool), diagonal=1)
        -> (context_size, context_size)

    Mask_Sliced = Mask[:seq_length, :seq_length]  # seq_length will smaller than context_size
        -> (seq_length, seq_length)

    Masked_Scores = Scores.masked_fill(Mask_Sliced, -inf)
        -> (batch_size, seq_length, seq_length)

    Attention Weights:
    A = softmax(Masked_Scores dim=-1) then dropout
        -> (batch_size, seq_length, seq_length)

    Attention Output:
    Output = A @ V
        -> (batch_size, seq_length, value_dim)

MutiHeadAttention:

    x  -> (batch_size, seq_length, embedding_dim)

    #          ┌─────────┐
    #          │  Wq_H0  │
    #          ├─────────┤
    #          │  Wq_H1  │
    # Wq   =   ├─────────┤
    #          │  Wq_H2  │
    #          ├─────────┤
    #          │  Wq_H3  │
    #          └─────────┘

    Wq -> (head_num, query_dim, embedding_dim) -> (total_query_dim, embedding_dim)
    Wk -> (total_key_dim, embedding_dim)
    Wv -> (total_value_dim, embedding_dim)

    # nn.Linear applies the same transformation
    # independently to every Token in every Batch.

    Q = W_query(x) -> (batch_size, seq_length, total_query_dim) -> (batch_size, seq_length, head_num, query_dim)
    K = W_key(x)   -> (batch_size, seq_length, total_key_dim)
    V = W_value(x) -> (batch_size, seq_length, total_value_dim)

    Attention Scores:
    DO RESHAPE TO QK TO SEPARATE DIFFERENT HEAD, THEN TRANSPOSE TO ENABLE

    Q_Reshape -> (batch_size, seq_length, head_num, query_dim)
    same to K and V

    Q_Transposed = Q.transpose(1, 2)
        -> (batch_size, head_num, seq_length, query_dim)
    same to K and V

    K_transposed = K.transpose(-2, -1)
        -> (batch_size, head_num, key_dim, seq_length)

    Scores = Q @ K_transposed / sqrt(qkv_dim_for_each_head)
        -> (batch_size, head_num, seq_length, seq_length) - row : one token as query ; column : one token as key

    Causal Mask:
    Every HEAD share the same Mask
    Mask = triu(ones(context_size, context_size, dtype=torch.bool), diagonal=1)
        -> (context_size, context_size)

    Mask_Sliced = Mask[:seq_length, :seq_length]  # seq_length will smaller than context_size
        -> (seq_length, seq_length)

    Masked_Scores = Scores.masked_fill(Mask_Sliced, -inf)
        -> (batch_size, head_num, seq_length, seq_length)

    Attention Weights:
    A = softmax(Masked_Scores dim=-1) then dropout
        -> (batch_size, head_num, seq_length, seq_length)

    Attention Output:
    Output = A @ V
        -> (batch_size, head_num, seq_length, value_dim)

    # Merge Heads:
    Output = Output.transpose(1, 2)
        -> (batch_size, seq_length, head_num, value_dim)

    Output = Output.contiguous().reshape(batch_size, seq_length, total_value_dim) -> total_value_dim = head_num * value_dim

    Projection for merge:
    Output_proj -> (total_qkv_dim, embedding_dim)

    Output = Output @ Output.proj
        -> (batch_size, seq_length, embedding_dim)

"""

import torch
from torch import nn

# for study purpose only, not in trainning
class CausalAttention(nn.Module):
    def __init__(self, qkv_dim, embedding_dim, context_size):
        super(CausalAttention, self).__init__()
        self.W_query = nn.Linear(embedding_dim, qkv_dim, bias=False)
        self.W_key   = nn.Linear(embedding_dim, qkv_dim, bias=False)
        self.W_value = nn.Linear(embedding_dim, qkv_dim, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.register_buffer('mask', torch.triu(
            torch.ones(context_size, context_size,
            dtype=torch.bool),
            diagonal=1)
            )

    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape

        keys = self.W_key(x) # x @ W.T == (W @ x.T).T, x input here is already transposed, so shape (context_size, embedding_dim)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Q = Q_math.T  K = K_math.T
        # Q @ K.T == Q_math.T @ K_math.T.T == Q_math.T @ K.math == S.math
        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores.masked_fill(self.mask[:seq_length, :seq_length], -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        # the reason why use dropout after weight instead of before weight: want the expectation of multiple group of weights remain unchanged, without asking sum == 1(or will change the expectation), but the single value of weight can be changed by dropout 1 / (1-p), because this fit the design to add some random noise; why expectation? the scale changed seems like changing the feature activation, but it actually made the activation remain unchanged, since we finally don't need x, but Wx.
        # 1 / (1 - p) comes from expectation of each element in Wx (assume that W is const)
        attention_weights = self.dropout(attention_weights)

        output = attention_weights @ values
        return output

class MutiHeadAttention(nn.Module):
    def __init__(self, head_num, qkv_dim, embedding_dim, context_size, drop_rate=0.1):
        super(MutiHeadAttention, self).__init__()
        self.qkv_dim = qkv_dim
        self.head_num = head_num
        total_qkv_dim = qkv_dim * head_num

        self.W_query = nn.Linear(embedding_dim, total_qkv_dim, bias=False)
        self.W_key = nn.Linear(embedding_dim, total_qkv_dim, bias=False)
        self.W_value = nn.Linear(embedding_dim, total_qkv_dim, bias=False)

        self.output_proj = nn.Linear(total_qkv_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer('mask', torch.triu(
            torch.ones(context_size, context_size,
            dtype=torch.bool),
            diagonal=1)
            )

    def forward(self, x):
        batch_size, seq_length, embedding_dim = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        queries_per_head = queries.reshape(batch_size, seq_length, self.head_num, self.qkv_dim).transpose(1, 2)
        keys_per_head = keys.reshape(batch_size, seq_length, self.head_num, self.qkv_dim).transpose(1, 2)
        values_per_head = values.reshape(batch_size, seq_length, self.head_num, self.qkv_dim).transpose(1, 2)

        attention_scores = queries_per_head @ keys_per_head.transpose(-2, -1)
        attention_scores = attention_scores.masked_fill(self.mask[:seq_length, :seq_length], -torch.inf)


        attention_weights = torch.softmax(attention_scores / self.qkv_dim ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = (attention_weights @ values_per_head) \
            .transpose(1, 2) \
            .reshape(batch_size, seq_length, self.head_num * self.qkv_dim)

        output = self.output_proj(output)
        return output
