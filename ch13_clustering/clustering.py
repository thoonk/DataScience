from scratch.linear_algebra import Vector, distance
import random
from matplotlib import pyplot as plt
from typing import NamedTuple, Union, Callable, List, Tuple


def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


print(num_differences([1, 2, 3], [2, 1, 3]))
print(num_differences([1, 2], [1, 2]))

from scratch.linear_algebra import vector_mean


# cluster별 평균값(중앙값)을 리스트로 만들어 리턴
def cluster_means(k: int, inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:  # 클러스터 소속을 표시하는 번호 리스트
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # if a cluster is empty, just use a random point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]


# i = [[1], [2], [5], [6]]  # 1차원
# a = [0, 1, 1, 0]
# print(cluster_means(2, i, a))

import itertools
import tqdm
from scratch.linear_algebra import squared_distance


class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k  # number of clusters
        self.means = None

    #  각 input 데이터 하나를 받았을때 각 센트로이드 거리 구해서 가장 적은 거리의 번호를 리턴함
    def classify(self, input: Vector) -> int:
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute menas and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignmnets changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    print(assignments)  # 확인
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")


# km = KMeans(2)
# km.train(i)
# print(km.means)

def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    # finds the total squared error from k-means clustering the inputs
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(input) for input in inputs]

    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))


class Leaf(NamedTuple):
    value: Vector


leaf1 = Leaf([10, 20])
leaf2 = Leaf([30, -15])


class Merged(NamedTuple):
    children: tuple
    order: int


merged = Merged((leaf1, leaf2), order=1)
Cluster = Union[Leaf, Merged]


# cluster에 포함된 모든 원소 값을 리스트로 리턴
def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]


c1 = Merged((leaf1, merged), 2)
# print(get_values(merged)) #[[10, 20], [30, -15]]
print(get_values(c1))  # [[10, 20], [10, 20], [30, -15]]


def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable = min) -> float:
    # compute all the pairwise distances between cluster1 and cluster2
    # and apply the aggregation function _distance_agg_ to the resulting list
    return distance_agg([distance(v1, v2)]
                        for v1 in get_values(cluster1)
                        for v2 in get_values(cluster2))


def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float('inf')  # was never merged
    else:
        return cluster.order


def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children


def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable = min) -> Cluster:
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)

    # as long as we have more than one cluster left..
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min(((cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]),
                     key=pair_distance)
        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]
        # merge them, using merge_order = # of clusters left
        merged_cluster = Merged((c1, c2), order=len(clusters))
        # and add their merge
        clusters.append(merged_cluster)

    return clusters[0]  # 최상위 레밸 클러스터 리턴 // 하나 남은 클러스터


def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> List[Cluster]:
    # 최종 하나의 클러스터를 의미
    clusters = [base_cluster]

    # 지정한 수까지 반복
    while len(clusters) < num_clusters:
        next_cluster = min(clusters, key=get_merge_order)
        # 리스트에서 제거
        clusters = [c for c in clusters if c != next_cluster]
        # 리스트에 그 클러스터의 하위 자식들 추가함
        clusters.extend(get_children(next_cluster))
    # 지정된 수의 클러스터가 나오면 리턴함
    return clusters


def main():
    inputs: List[List[float]] = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13],
                                 [-46, 5], [-34, -1], [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19],
                                 [-41, 8], [-11, -6], [-25, -9], [-18, -3]]

    random.seed(12)  # so you get the same results as me
    clusterer = KMeans(k=3)
    clusterer.train(inputs)
    means = sorted(clusterer.means)  # sort for the unit test // means만 남음
    print("means when k=3" + str(means))

    random.seed(0)
    clusterer = KMeans(k=2)
    clusterer.train(inputs)
    means = sorted(clusterer.means)
    print("means when k=2" + str(means))

    # plot from 1 up to len(inputs) clusters // k 값 선택하기
    # ks = range(1, len(inputs) + 1)
    # errors = [squared_clustering_errors(inputs, k) for k in ks]
    #
    # plt.plot(ks, errors)
    # plt.xticks(ks)
    # plt.xlabel("k")
    # plt.ylabel("total squared error")
    # plt.title("Total Error vs. # of Clusters")
    # plt.show()
    #
    # plt.savefig('im/total_error_vs_num_clusters')
    # plt.gca().clear()

    # --------------------RGB 군집화---------------------
    # image_path = r"girl_with_book.jpg"
    # import matplotlib.image as mpimg
    # img = mpimg.imread(image_path) / 256  # rescale to between 0 and 1
    # # print(img[:1])
    #
    # pixels = [pixel.tolist() for row in img for pixel in row]
    #
    # clusterer = KMeans(5)
    # clusterer.train(pixels)
    #
    # def recolor(pixel: Vector) -> Vector:
    #     cluster = clusterer.classify(pixel)
    #     return clusterer.means[cluster]
    #
    # new_img = [[recolor(pixel) for pixel in row] for row in img]
    #
    # plt.close()
    # plt.imshow(new_img)
    # plt.axis('off')
    # # plt.show()
    #
    # plt.savefig('im/recolored_girl_with_book.jpg')
    # plt.gca().clear()

    # base_cluster = bottom_up_cluster(inputs)
    # # print(base_cluster)
    # three_cluster = [get_values(cluster)
    #                  for cluster in generate_clusters(base_cluster, 3)]
    #
    # for i, cluster, marker, color in zip([1, 2, 3],
    #                                      three_cluster,
    #                                      ['D', 'o', '*'],
    #                                      ['r', 'g', 'b']):
    #     xs, ys = zip(*cluster)
    #     plt.scatter(xs, ys, color=color, marker=marker)
    #
    #     x, y = vector_mean(cluster)
    #     plt.plot(x, y, marker='$' + str(i) + '$', color='black')
    #
    # plt.title("User Locations -- 3 Bottom-Up clusters, Min")
    # plt.xlabel("blocks east of city center")
    # plt.ylabel("blocks north of city center")
    # # plt.show()

    base_cluster_max = bottom_up_cluster(inputs, max);
    three_cluster_max = [get_values(cluster)
                         for cluster in generate_clusters(base_cluster_max, 3)]

    for i, cluster, marker, color in zip([1, 2, 3],
                                         three_cluster_max,
                                         ['D', 'o', '*'],
                                         ['r', 'g', 'b']):
        xs, ys = zip(*cluster)
        plt.scatter(xs, ys, color=color, marker=marker)

        x, y = vector_mean(cluster)
        plt.plot(x, y, marker='$' + str(i) + '$', color='black')

    plt.title("User Locations -- 3 Bottom-Up clusters, Min")
    plt.xlabel("blocks east of city center")
    plt.ylabel("blocks north of city center")
    plt.show()


if __name__ == "__main__": main()
