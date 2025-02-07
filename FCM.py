import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def interpolate_track(track, num_segments=20):
    # 原始路径点数量
    num_original_points = len(track)
    # 计算原始路径各段距离
    distances = np.linalg.norm(np.diff(track, axis=0), axis=1)
    total_distance = np.sum(distances)
    segment_length = total_distance / num_segments

    interpolated_track = np.zeros((num_segments + 1, 2))
    interpolated_track[0] = track[0]
    interpolated_track[-1] = track[-1]

    current_distance = 0
    segment_index = 1
    for i in range(num_original_points - 1):
        if current_distance + distances[i] >= segment_length * segment_index:
            while current_distance + distances[i] >= segment_length * segment_index:
                ratio = (segment_length * segment_index - current_distance) / distances[i]
                interpolated_track[segment_index] = track[i] + ratio * (track[i + 1] - track[i])
                segment_index += 1
                if segment_index == num_segments:
                    break
            current_distance = 0
        else:
            current_distance += distances[i]
    return interpolated_track


def preprocess_tracks(tracks, num_segments=20):
    interpolated_tracks = []
    for track in tracks:
        interpolated_track = interpolate_track(track, num_segments)
        interpolated_tracks.append(interpolated_track.flatten())
    return np.array(interpolated_tracks)


def determine_optimal_cluster_number(data, max_clusters=12):
    partition_coefficients = []
    partition_indices = []
    separation_indices = []
    dunn_indices = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, max_iter=300, n_init=10, init='k-means++')
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 计算分区系数
        partition_coefficient = np.sum([np.sum(np.square(kmeans.predict([point]))) for point in data]) / len(data)
        partition_coefficients.append(partition_coefficient)

        # 计算分区指数（简化计算，实际可能需要更精确方法）
        compactness = np.sum([np.linalg.norm(point - centers[labels[i]]) for i, point in enumerate(data)])
        separation = np.sum([np.min([np.linalg.norm(centers[i] - centers[j]) for j in range(k) if j != i]) for i in range(k)])
        partition_index = compactness / separation if separation != 0 else np.inf
        partition_indices.append(partition_index)

        # 计算分离指数（简化计算，实际可能需要更精确方法）
        separation_index = compactness / np.min([np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(k) if j != i]) if np.min([np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(k) if j != i]) != 0 else np.inf
        separation_indices.append(separation_index)

        # 计算邓恩指数（简化计算，实际可能需要更精确方法）
        min_inter_cluster_distances = []
        max_intra_cluster_distances = []
        for i in range(k):
            cluster_points = data[labels == i]
            intra_cluster_distances = [np.linalg.norm(point1 - point2) for point1 in cluster_points for point2 in cluster_points]
            max_intra_cluster_distances.append(np.max(intra_cluster_distances))
            inter_cluster_distances = []
            for j in range(k):
                if j != i:
                    other_cluster_points = data[labels == j]
                    for point1 in cluster_points:
                        for point2 in other_cluster_points:
                            inter_cluster_distances.append(np.linalg.norm(point1 - point2))
            min_inter_cluster_distances.append(np.min(inter_cluster_distances))
        dunn_index = np.min(min_inter_cluster_distances) / np.max(max_intra_cluster_distances) if np.max(max_intra_cluster_distances) != 0 else np.inf
        dunn_indices.append(dunn_index)

    # 根据指标选择最佳聚类数（简单示例，实际应综合判断）
    optimal_k = np.argmin(separation_indices) + 2
    return optimal_k


def plot_validity_measures(max_clusters=12):
    # 假设这里已经有预处理好的数据
    data = np.random.rand(100, 40)
    partition_coefficients = []
    partition_indices = []
    separation_indices = []
    dunn_indices = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, max_iter=300, n_init=10, init='k-means++')
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 计算分区系数
        partition_coefficient = np.sum([np.sum(np.square(kmeans.predict([point]))) for point in data]) / len(data)
        partition_coefficients.append(partition_coefficient)

        # 计算分区指数（简化计算，实际可能需要更精确方法）
        compactness = np.sum([np.linalg.norm(point - centers[labels[i]]) for i, point in enumerate(data)])
        separation = np.sum([np.min([np.linalg.norm(centers[i] - centers[j]) for j in range(k) if j != i]) for i in range(k)])
        partition_index = compactness / separation if separation != 0 else np.inf
        partition_indices.append(partition_index)

        # 计算分离指数（简化计算，实际可能需要更精确方法）
        separation_index = compactness / np.min([np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(k) if j != i]) if np.min([np.linalg.norm(centers[i] - centers[j]) for i in range(k) for j in range(k) if j != i]) != 0 else np.inf
        separation_indices.append(separation_index)

        # 计算邓恩指数（简化计算，实际可能需要更精确方法）
        min_inter_cluster_distances = []
        max_intra_cluster_distances = []
        for i in range(k):
            cluster_points = data[labels == i]
            intra_cluster_distances = [np.linalg.norm(point1 - point2) for point1 in cluster_points for point2 in cluster_points]
            max_intra_cluster_distances.append(np.max(intra_cluster_distances))
            inter_cluster_distances = []
            for j in range(k):
                if j != i:
                    other_cluster_points = data[labels == j]
                    for point1 in cluster_points:
                        for point2 in other_cluster_points:
                            inter_cluster_distances.append(np.linalg.norm(point1 - point2))
            min_inter_cluster_distances.append(np.min(inter_cluster_distances))
        dunn_index = np.min(min_inter_cluster_distances) / np.max(max_intra_cluster_distances) if np.max(max_intra_cluster_distances) != 0 else np.inf
        dunn_indices.append(dunn_index)

    plt.figure(figsize=(12, 9))

    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_clusters + 1), partition_coefficients)
    plt.title('Partition Coefficient')
    plt.xlabel('Number of clusters')
    plt.ylabel('Partition Coefficient Value')

    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_clusters + 1), partition_indices)
    plt.title('Partition Index')
    plt.xlabel('Number of clusters')
    plt.ylabel('Partition Index Value')

    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_clusters + 1), separation_indices)
    plt.title('Separation Index')
    plt.xlabel('Number of clusters')
    plt.ylabel('Separation Index Value')

    plt.subplot(2, 2, 4)
    plt.plot(range(2, max_clusters + 1), dunn_indices)
    plt.title('Dunn Index')
    plt.xlabel('Number of clusters')
    plt.ylabel('Dunn Index Value')

    plt.tight_layout()
    plt.show()

def cluster_tracks(tracks, num_clusters):
    preprocessed_tracks = preprocess_tracks(tracks)
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, n_init=10, init='k-means++')
    kmeans.fit(preprocessed_tracks)
    labels = kmeans.labels_
    return labels

# 示例数据（需替换为真实台风路径数据，格式为：[ [ [lat1, lon1], [lat2, lon2],... ], [ [lat1, lon1],... ],... ] ）
# 这里假设已经从文件或其他方式获取到了tracks数据
# tracks = get_real_tracks_data()
# 示例数据
tracks = [np.random.rand(10, 2) for _ in range(50)]
# 预处理数据
preprocessed_tracks = preprocess_tracks(tracks)
# 确定最佳聚类数
optimal_num_clusters = determine_optimal_cluster_number(preprocessed_tracks)
# 聚类
labels = cluster_tracks(tracks, optimal_num_clusters)
# 输出聚类结果（简单打印示例）
for i, label in enumerate(labels):
    print(f"Track {i} belongs to cluster {label}")
# 绘制有效性度量图
plot_validity_measures()