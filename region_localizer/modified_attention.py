"""Изменённый MultiheadAttention для локализации фрагмента на карте."""


from typing import List, Optional, Tuple
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import linear, softmax, dropout


# source - torch.nn.functional 4727
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation,
    using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
            For self-attention, these are typically the same tensor;
            for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.)
            Regardless, q, k and v must share a common embedding dimension;
            otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor.
            Weights are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single
            tensor in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory
            # coalescing and keeping same order as chunk()
            proj = (proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2)
                    .squeeze(-2).contiguous())
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory
            # coalescing and keeping same order as chunk()
            kv_proj = (kv_proj.unflatten(-1, (2, E)).unsqueeze(0)
                       .transpose(0, -2).squeeze(-2).contiguous())
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# source - torch.nn.functional 5017
def modified_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    average_attn_weights: bool = True
) -> Tuple[Tensor, Tensor]:
    """Изменённый multihead attention forward.

    Главное изменение - конечное матричное умножение attn_weights на value
    было заменено на обычное линейное.
    `torch.bmm(attn_output_weights, v) -> attn_output_weights * v`
    что в данном контексте позволяет умножить веса внимания на логиты от карты
    и получить размер `(b, n_pieces, embed)` вместо `(b, 1, embed)`.
    Для этого была использована ветвь с `need_weights == True`, а другая
    вырезана.

    Также были вырезаны вся функциональность с add_zero_attn, масками,
    раздельными весами, статичными k и v, is_causal за ненадобностью
    в данной реализации.

    Parameters
    ----------
    query : Tensor
        Тензор размером (b, 1, embed).
    key : Tensor
        Тензор размером (b, n_pieces, embed).
    value : Tensor
        Тензор размером (b, n_pieces, embed).
    embed_dim_to_check : int
        Ожидаемый embed_dim для проверки.
    num_heads : int
        Количество голов. Должно быть кратно embed_dim.
    in_proj_weight : Optional[Tensor]
        Веса для входной проекции q, k, v.
        Передаются одной матрицей и затем разделяются.
    in_proj_bias : Optional[Tensor]
        Смещения для in_proj_weight.
    bias_k : Optional[Tensor]
        Отдельное смещение для key.
    bias_v : Optional[Tensor]
        Отдельное смещение для value.
    dropout_p : float
        Вероятность dropout.
    out_proj_weight : Tensor
        Веса для выходной проекции результата.
    out_proj_bias : Optional[Tensor]
        Смещения для out_proj_weight.
    training : bool, optional
        Флаг обучения. Выключает dropout если `False`. По умолчанию `True`.
    average_attn_weights : bool, optional
        Флаг усреднения весов внимания вдоль оси голов. По умолчанию `True`.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Взвешенная карта размером `(b, n_pieces, embed)` и веса внимания
        размером `(b, num_heads, 1, n_pieces)` если `average_attn_weights`
        равен `False` и `(b, 1, n_pieces)` при `True`.
    """
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError('Query, key and value must be 3-d.')

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if embed_dim != embed_dim_to_check:
        raise ValueError(
            f'Was expecting embedding dimension of {embed_dim_to_check}, '
            f'but got {embed_dim}')

    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(
            f'Embed_dim {embed_dim} not divisible by num_heads {num_heads}')

    if key.shape != value.shape:
        raise ValueError(
            f'Key shape {key.shape} does not match '
            f'value shape {value.shape}')
    #
    # compute in-projection
    #
    if in_proj_weight is None:
        raise ValueError(
            'Use_separate_proj_weight is False but in_proj_weight is None')
    q, k, v = _in_projection_packed(
        query, key, value, in_proj_weight, in_proj_bias)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
    else:
        if bias_k is not None or bias_v is not None:
            raise ValueError(
                'If one of "bias_v" or "bias_v" is provided '
                'then they both must be provided.')

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)

    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    ####################################################################
    # Изменение
    attn_output = attn_output_weights.permute(0, 2, 1) * v

    attn_output = attn_output.transpose(0, 1).contiguous().view(
        k.shape[1] * bsz, embed_dim)
    # b * head, hw, embed // head -> hw * b, embed

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(k.shape[1], bsz, attn_output.size(1))

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(
        bsz, num_heads, src_len, tgt_len)
    ####################################################################
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    return attn_output, attn_output_weights


class ModifiedAttention(nn.MultiheadAttention):
    def __init__(
        self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False,
        kdim=None, vdim=None, batch_first=False, device=None, dtype=None
    ) -> None:
        # Cut out zero attention
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, None, kdim, vdim,
            batch_first, device, dtype)
        
    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Изменённый multihead attention слой.

        Переопределён forward метод и вырезана часть функционала.
        Главное изменение - конечное матричное умножение attn_weights на value
        было заменено на обычное линейное.
        `torch.bmm(attn_output_weights, v) -> attn_output_weights * v`
        что в данном контексте позволяет умножить веса внимания на логиты от
        карты и получить размер `(b, n_pieces, embed)` вместо `(b, 1, embed)`.
        Подробнее в описании функции `modified_multi_head_attention_forward`.

        Parameters
        ----------
        query : Tensor
            Тензор размера `(b, 1, embed)`.
        key : Tensor
            Тензор размера `(b, n_pieces, embed)`.
        value : Tensor
            Тензор размера `(b, n_pieces, embed)`.
        average_attn_weights : bool, optional
            Флаг усреднения весов внимания вдоль оси голов.
            По умолчанию `True`.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Взвешенная карта размером `(b, n_pieces, embed)` и веса внимания
            размером `(b, num_heads, 1, n_pieces)` если `average_attn_weights`
            равен `False` и `(b, 1, n_pieces)` при `True`.
        """
        if self.batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0)
                                     for x in (query, key, value)]

        attn_output, attn_output_weights = (
            modified_multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.dropout,
                self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                average_attn_weights=average_attn_weights))
        
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
