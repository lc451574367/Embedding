# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:04:11 2024

@author: 45157
"""

from feature.get_vector import *

import numpy as np
import requests

def Cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def Cosine_similarity_matrix(vecs1, vecs2):
    # 计算vecs1中每个向量的范数
    norms1 = np.linalg.norm(vecs1, axis=1)
    # 计算vecs2中每个向量的范数
    norms2 = np.linalg.norm(vecs2, axis=1)
    # 计算点积矩阵
    dot_product = np.dot(np.array(vecs1), np.array(vecs2).T)
    # 计算余弦相似度矩阵
    return dot_product / np.outer(norms1, norms2)

def fetch_conceptnet_relationships(word1, word2, relationship):
    # 创建一个以单词为索引和列名的空 DataFrame
    results_df = pd.DataFrame(index=word1, columns=word2)

    # 对每一对单词进行查询
    for w1 in word1:
        for w2 in word2:
            if w1 == w2:
                results_df.at[w2, w2] = 1.0  # 自己与自己的关系权重设为1
            else:
                try:
                    # 构建 API 请求的 URL
                    url = f"http://api.conceptnet.io/query?node=/c/en/{w1}&other=/c/en/{w2}&rel=/r/{relationship}"
                    response = requests.get(url)
                    response.raise_for_status()  # 如果响应状态码不是 200，则抛出 HTTPError 异常
                    data = response.json()

                    # 解析结果，计算权重（这里简单地使用返回的边数作为权重）
                    weight = sum(edge['weight'] for edge in data['edges'])
                    results_df.at[w1, w2] = weight
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error occurred for {w1}-{w2}: {e}")
                    results_df.at[w1, w2] = 0  # 在出错时赋予权重0
                except requests.exceptions.RequestException as e:
                    print(f"Request error occurred for {w1}-{w2}: {e}")
                    results_df.at[w1, w2] = 0  # 在出错时赋予权重0
                except KeyError:
                    print(f"Unexpected response data structure for {w1}-{w2}.")
                    results_df.at[w1, w2] = 0  # 如果数据结构不如预期，赋予权重0
                except Exception as e:
                    print(f"An unexpected error occurred for {w1}-{w2}: {e}")
                    results_df.at[w1, w2] = 0  # 捕获其他所有未预见的异常

    return results_df

# # 示例使用
# word1 = ['happy']
# word2 = ['apple', 'banana', 'fruit', 'orange']
# relationship = 'RelatedTo'
# relationship_df = fetch_conceptnet_relationships(word1,word2, relationship)
# print(relationship_df)