"""
练习 1: 图论基础
================

本练习包含3道题目，由浅入深：
- 初级：图的基本表示和性质
- 中级：图的遍历和连通性
- 高级：最短路径和图算法

学习目标：
- 理解图的邻接矩阵和邻接表表示
- 掌握图的遍历算法 (DFS, BFS)
- 实现经典图算法 (Dijkstra, PageRank)
- 为图神经网络(GNN)打基础

应用领域：
- 图神经网络 (Graph Neural Networks)
- 社交网络分析
- 推荐系统
- 知识图谱

参考教材：
- West - Introduction to Graph Theory
- Bondy & Murty - Graph Theory
- Newman - Networks: An Introduction
"""

import numpy as np
from collections import deque, defaultdict
import sys
sys.path.append('src')
from validator import Validator, create_test_decorator


# ============================================================================
# 初级题目：图的基本表示和性质
# ============================================================================

def adjacency_matrix_to_list(adj_matrix):
    """
    初级 - 邻接矩阵转邻接表

    图的两种主要表示方式：
    1. 邻接矩阵: n×n矩阵，A[i,j]=1表示有边i→j
    2. 邻接表: 字典，存储每个节点的邻居列表

    参数:
        adj_matrix: n×n numpy数组，邻接矩阵

    返回:
        邻接表字典 {node: [neighbors]}
    """
    # TODO: 实现邻接矩阵到邻接表的转换
    n = adj_matrix.shape[0]
    adj_list = defaultdict(list)

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                adj_list[i].append(j)

    return dict(adj_list)


def graph_degree(adj_matrix):
    """
    初级 - 计算图的度序列

    度(degree): 一个节点连接的边数
    - 无向图: deg(v) = 与v相连的边数
    - 有向图: in-degree (入度) + out-degree (出度)

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        度序列数组
    """
    # TODO: 计算每个节点的度
    # 对于无向图: 行和即为度
    # 对于有向图: 这里计算出度(out-degree)
    degrees = np.sum(adj_matrix, axis=1)
    return degrees


def is_connected_graph(adj_matrix):
    """
    初级 - 判断无向图是否连通

    连通图: 任意两个节点之间都存在路径
    方法: 从任意节点开始BFS/DFS，看是否能访问所有节点

    参数:
        adj_matrix: n×n对称邻接矩阵（无向图）

    返回:
        布尔值，是否连通
    """
    # TODO: 实现连通性判断
    n = adj_matrix.shape[0]
    if n == 0:
        return True

    visited = set()
    queue = deque([0])  # 从节点0开始BFS
    visited.add(0)

    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if adj_matrix[node, neighbor] != 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n


# ============================================================================
# 中级题目：图的遍历和连通性
# ============================================================================

def depth_first_search(adj_list, start):
    """
    中级 - 深度优先搜索 (DFS)

    DFS策略: 尽可能深地搜索图的分支
    应用: 拓扑排序、连通分量、环检测

    参数:
        adj_list: 邻接表字典
        start: 起始节点

    返回:
        DFS访问顺序列表
    """
    # TODO: 实现DFS
    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        order.append(node)

        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor)

    dfs(start)
    return order


def breadth_first_search(adj_list, start):
    """
    中级 - 广度优先搜索 (BFS)

    BFS策略: 按层次遍历，先访问距离近的节点
    应用: 最短路径、层次遍历、GNN消息传递

    参数:
        adj_list: 邻接表字典
        start: 起始节点

    返回:
        BFS访问顺序列表
    """
    # TODO: 实现BFS
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return order


def find_connected_components(adj_matrix):
    """
    中级 - 寻找无向图的连通分量

    连通分量: 最大的连通子图
    方法: 对每个未访问节点运行DFS/BFS

    参数:
        adj_matrix: n×n对称邻接矩阵

    返回:
        连通分量列表，每个分量是节点集合
    """
    # TODO: 实现连通分量查找
    n = adj_matrix.shape[0]
    visited = set()
    components = []

    for start in range(n):
        if start not in visited:
            # BFS找到一个连通分量
            component = set()
            queue = deque([start])
            visited.add(start)
            component.add(start)

            while queue:
                node = queue.popleft()
                for neighbor in range(n):
                    if adj_matrix[node, neighbor] != 0 and neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(list(component))

    return components


def graph_laplacian(adj_matrix):
    """
    中级 - 计算图拉普拉斯矩阵

    拉普拉斯矩阵: L = D - A
    - D: 度矩阵（对角矩阵，D[i,i] = deg(i)）
    - A: 邻接矩阵

    应用: 谱聚类、图卷积神经网络(GCN)

    参数:
        adj_matrix: n×n邻接矩阵

    返回:
        拉普拉斯矩阵
    """
    # TODO: 实现拉普拉斯矩阵
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix
    return L


# ============================================================================
# 高级题目：最短路径和图算法
# ============================================================================

def dijkstra_shortest_path(adj_matrix, start, end):
    """
    高级 - Dijkstra最短路径算法

    计算加权图中从start到end的最短路径
    时间复杂度: O(V²) 或 O((V+E)logV) 使用优先队列

    参数:
        adj_matrix: n×n权重矩阵，0表示无边，>0表示边权重
        start: 起始节点
        end: 目标节点

    返回:
        (距离, 路径) 元组
    """
    # TODO: 实现Dijkstra算法
    n = adj_matrix.shape[0]
    distances = np.full(n, np.inf)
    distances[start] = 0
    visited = set()
    previous = {i: None for i in range(n)}

    for _ in range(n):
        # 找到未访问的最小距离节点
        min_dist = np.inf
        min_node = -1
        for node in range(n):
            if node not in visited and distances[node] < min_dist:
                min_dist = distances[node]
                min_node = node

        if min_node == -1:
            break

        visited.add(min_node)

        # 更新邻居距离
        for neighbor in range(n):
            if adj_matrix[min_node, neighbor] > 0:
                new_dist = distances[min_node] + adj_matrix[min_node, neighbor]
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = min_node

    # 重建路径
    if distances[end] == np.inf:
        return (np.inf, [])

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    return (distances[end], path)


def pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """
    高级 - PageRank算法

    Google的网页排名算法，也用于图节点重要性评估

    迭代公式:
    PR(v) = (1-d)/N + d * Σ(PR(u)/L(u))

    其中u是指向v的节点，L(u)是u的出度

    参数:
        adj_matrix: n×n邻接矩阵
        damping: 阻尼因子(通常0.85)
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        PageRank分数数组
    """
    # TODO: 实现PageRank
    n = adj_matrix.shape[0]

    # 初始化PageRank (均匀分布)
    pr = np.ones(n) / n

    # 计算出度
    out_degree = np.sum(adj_matrix, axis=1)
    out_degree[out_degree == 0] = 1  # 避免除以0

    for iteration in range(max_iter):
        pr_new = np.zeros(n)

        for i in range(n):
            # 找到所有指向i的节点
            incoming_sum = 0
            for j in range(n):
                if adj_matrix[j, i] != 0:
                    incoming_sum += pr[j] / out_degree[j]

            pr_new[i] = (1 - damping) / n + damping * incoming_sum

        # 检查收敛
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break

        pr = pr_new

    return pr


def floyd_warshall(adj_matrix):
    """
    高级 - Floyd-Warshall全对最短路径算法

    计算所有节点对之间的最短路径
    动态规划方法，时间复杂度O(V³)

    参数:
        adj_matrix: n×n权重矩阵，0表示无边

    返回:
        距离矩阵 dist[i][j] = i到j的最短距离
    """
    # TODO: 实现Floyd-Warshall算法
    n = adj_matrix.shape[0]

    # 初始化距离矩阵
    dist = np.copy(adj_matrix).astype(float)

    # 0表示无边，转换为无穷大
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)  # 自己到自己距离为0

    # 动态规划
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def minimum_spanning_tree_prim(adj_matrix):
    """
    高级 - Prim最小生成树算法

    最小生成树(MST): 连接所有节点的最小权重树
    应用: 网络设计、聚类

    参数:
        adj_matrix: n×n对称权重矩阵

    返回:
        MST的边列表 [(u, v, weight), ...]
    """
    # TODO: 实现Prim算法
    n = adj_matrix.shape[0]
    visited = set([0])  # 从节点0开始
    mst_edges = []

    while len(visited) < n:
        min_edge = None
        min_weight = np.inf

        # 找到连接visited和未visited节点的最小权重边
        for u in visited:
            for v in range(n):
                if v not in visited and adj_matrix[u, v] > 0:
                    if adj_matrix[u, v] < min_weight:
                        min_weight = adj_matrix[u, v]
                        min_edge = (u, v, adj_matrix[u, v])

        if min_edge is None:
            break  # 图不连通

        u, v, w = min_edge
        visited.add(v)
        mst_edges.append(min_edge)

    return mst_edges


# ============================================================================
# 测试函数
# ============================================================================

@create_test_decorator
def test():
    """测试函数"""
    v = Validator()
    results = []

    print("\n" + "="*60)
    print("初级题目：图的基本表示和性质")
    print("="*60)

    # 创建测试图
    # 0 -- 1 -- 2
    # |         |
    # 3 ------- 4
    adj_matrix = np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ])

    print("\n测试 1.1: 邻接矩阵转邻接表")
    adj_list = adjacency_matrix_to_list(adj_matrix)
    expected_list = {0: [1, 3], 1: [0, 2], 2: [1, 4], 3: [0, 4], 4: [2, 3]}
    # 验证邻接表
    correct = all(set(adj_list.get(k, [])) == set(v) for k, v in expected_list.items())
    results.append(correct)
    print(f"  邻接表转换: {'✓ 通过' if correct else '✗ 失败'}")

    print("\n测试 1.2: 图的度序列")
    degrees = graph_degree(adj_matrix)
    expected_degrees = np.array([2, 2, 2, 2, 2])
    results.append(v.assert_array_equal(degrees, expected_degrees, name="度序列"))

    print("\n测试 1.3: 图的连通性")
    is_conn = is_connected_graph(adj_matrix)
    results.append(is_conn)
    print(f"  连通性检测: {'✓ 通过（图连通）' if is_conn else '✗ 失败'}")

    print("\n" + "="*60)
    print("中级题目：图的遍历和连通性")
    print("="*60)

    print("\n测试 2.1: 深度优先搜索")
    dfs_order = depth_first_search(adj_list, 0)
    # DFS顺序取决于实现，但应该访问所有节点
    results.append(len(dfs_order) == 5 and set(dfs_order) == set(range(5)))
    print(f"  DFS顺序: {dfs_order}")
    print(f"  {'✓ 通过（访问所有节点）' if results[-1] else '✗ 失败'}")

    print("\n测试 2.2: 广度优先搜索")
    bfs_order = breadth_first_search(adj_list, 0)
    results.append(len(bfs_order) == 5 and set(bfs_order) == set(range(5)))
    print(f"  BFS顺序: {bfs_order}")
    print(f"  {'✓ 通过（访问所有节点）' if results[-1] else '✗ 失败'}")

    print("\n测试 2.3: 连通分量")
    components = find_connected_components(adj_matrix)
    results.append(len(components) == 1)  # 只有一个连通分量
    print(f"  连通分量数: {len(components)}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n测试 2.4: 图拉普拉斯矩阵")
    L = graph_laplacian(adj_matrix)
    # 拉普拉斯矩阵每行和为0
    row_sums = np.sum(L, axis=1)
    results.append(v.assert_array_equal(row_sums, np.zeros(5), name="拉普拉斯矩阵行和"))

    print("\n" + "="*60)
    print("高级题目：最短路径和图算法")
    print("="*60)

    # 创建加权图
    weighted_adj = np.array([
        [0, 4, 0, 0, 0],
        [4, 0, 2, 5, 0],
        [0, 2, 0, 0, 3],
        [0, 5, 0, 0, 1],
        [0, 0, 3, 1, 0]
    ])

    print("\n测试 3.1: Dijkstra最短路径")
    dist, path = dijkstra_shortest_path(weighted_adj, 0, 4)
    # 0->1->2->4 距离 = 4+2+3 = 9
    results.append(v.assert_close(dist, 9, name="Dijkstra距离"))
    print(f"  路径: {' -> '.join(map(str, path))}")

    print("\n测试 3.2: PageRank")
    # 创建有向图
    directed_adj = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    pr = pagerank(directed_adj)
    # PageRank和应该为1
    results.append(v.assert_close(np.sum(pr), 1.0, rtol=1e-4, name="PageRank和"))
    print(f"  PageRank分数: {pr}")

    print("\n测试 3.3: Floyd-Warshall全对最短路径")
    dist_matrix = floyd_warshall(weighted_adj)
    # 检查0到4的距离
    results.append(v.assert_close(dist_matrix[0, 4], 9, name="Floyd-Warshall距离"))

    print("\n测试 3.4: Prim最小生成树")
    mst = minimum_spanning_tree_prim(weighted_adj)
    # MST应该有n-1条边
    results.append(len(mst) == 4)
    total_weight = sum(w for _, _, w in mst)
    print(f"  MST边数: {len(mst)}, 总权重: {total_weight}")
    print(f"  {'✓ 通过' if results[-1] else '✗ 失败'}")

    print("\n" + "="*60)
    print(f"总体结果: {sum(results)}/{len(results)} 通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    test()
