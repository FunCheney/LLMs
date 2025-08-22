import torch
import torch.nn as nn
import math

'''计算注意力'''
def attention(query, key, value, dropout=None):
    '''
    假设输入的 q、k、v 是已经经过转化的词向量矩阵，也就是公式中的 Q、K、V
    :param query: 查询值矩阵
    :param key:   键值矩阵
    :param value: 真值举证
    :param dropout:  输出
    :return: 返回
    '''

    # 获取向量的维度，键的向量维度和值的向量维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

        # 采样
        # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

