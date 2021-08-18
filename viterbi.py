import numpy as np

# 斐波那契数列
def fib(n):
    resulits = list(range(n+1))
    for i in range(n+1):
        if i<2:
            resulits[i] = i
        else:
            resulits[i] = resulits[i-1] + resulits[i-2]
    return resulits[-1]


# 路径问题
#  A -> _ ..._
#  |
#  V
#  _
#  ...
#  _   ...    B
# A到B总共有多少条路径，一次只能往右或者往下走一步
def count_paths(m, n):
    results = [[1] * n] * m
    for i in range(1, m):
        for j in range(1, n):
            results[i][j] = results[i-1][j] + results[i][j-1]
    return results[-1][-1]


def viterbi_decode(nodes, trans):
    """
    Viterbi算法求最优路径
    其中 nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    # 获得输入状态序列的长度，以及观察标签的个数
    seq_len, num_labels = len(nodes), len(trans)
    # 简单起见，先不考虑发射概率，直接用起始0时刻的分数
    scores = nodes[0].reshape((-1, 1))

    paths = []
    # 递推求解上一时刻t-1到当前时刻t的最优
    for t in range(1, seq_len):
        # scores 表示起始0到t-1时刻的每个标签的最优分数
        scores_repeat = np.repeat(scores, num_labels, axis=1)
        # observe当前时刻t的每个标签的观测分数
        observe = nodes[t].reshape((1, -1))
        observe_repeat = np.repeat(observe, num_labels, axis=0)
        # 从t-1时刻到t时刻最优分数的计算，这里需要考虑转移分数trans
        M = scores_repeat + trans + observe_repeat
        # 寻找到t时刻的最优路径
        scores = np.max(M, axis=0).reshape((-1, 1))
        idxs = np.argmax(M, axis=0)
        # 路径保存
        paths.append(idxs.tolist())

    best_path = [0] * seq_len
    best_path[-1] = np.argmax(scores)
    # 最优路径回溯
    for i in range(seq_len - 2, -1, -1):
        idx = best_path[i + 1]
        best_path[i] = paths[i][idx]

    return best_path


if __name__ == '__main__':
    # result = fib(100)
    result = count_paths(7, 3)
    print(result)