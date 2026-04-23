import matplotlib.pyplot as plt
import numpy as np

# 维度
dimensions = [16, 32, 64, 128]

# 余弦相似度阈值
cosine_values = np.array([50, 60, 70, 80, 90])

# 生成不同维度的 F1 Scores 和 Trend Scores
data = {
    16: {
        "f1_both": np.array([71.8, 66.2, 58.1, 48.8, 37.9]),
        "f1_keywords": np.array([70.4, 64.1, 56, 46.9, 37.7]),
        "f1_metrics": np.array([79.7, 78.9, 75.9, 69, 53.3]),
        "precision_both": np.array([80.7, 84.9, 89.2, 94.3, 98.3]),
        "precision_keywords": np.array([82, 86, 88.8, 93.2, 98.3]),
        "precision_metrics": np.array([71, 73.9, 77.8, 84.2, 94.3]),
    },
    32: {
        "f1_both": np.array([64.6, 58, 50, 40.4, 32.6]),
        "f1_keywords": np.array([66.3, 58.5, 49.4, 39.9, 32]),
        "f1_metrics": np.array([79.7, 79.6, 78.6, 74.1, 58.9]),
        "precision_both": np.array([83.9, 88.9, 93.8, 97.6, 99.2]),
        "precision_keywords": np.array([84.3, 89.2, 93.4, 97.4, 99.1]),
        "precision_metrics": np.array([69.6, 72.1, 76.1, 81.8, 89.1]),
    },
    64: {
        "f1_both": np.array([61.3, 53.3, 44.7, 37, 30.3]),
        "f1_keywords": np.array([61, 52.7, 44.5, 36.9, 31.1]),
        "f1_metrics": np.array([80.1, 78.5, 72.8, 60.8, 43.2]),
        "precision_both": np.array([88.6, 93, 96.2, 98.6, 99.6]),
        "precision_keywords": np.array([91.7, 94.7, 97.1, 98.8, 99.3]),
        "precision_metrics": np.array([74.3, 78.7, 83.5, 87.9, 94.7]),
    },
    128: {
        "f1_both": np.array([58.4, 50.3, 42.2, 35.2, 29.8]),
        "f1_keywords": np.array([57.3, 49.6, 42.5, 35.8, 30.9]),
        "f1_metrics": np.array([79.6, 77.1, 71.5, 60.3, 42.8]),
        "precision_both": np.array([90.2, 93.7, 96.9, 98.9, 99.6]),
        "precision_keywords": np.array([89.6, 93, 95.9, 98.3, 99.1]),
        "precision_metrics": np.array([74.4, 78, 82.6, 88.3, 95.9]),
    },
}

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle("Performance Metrics Across Dimensions", fontsize=16)

# 颜色定义
colors = {'keywords': '#92c5de', 'metrics': '#f4a582', 'both': '#9970ab'}

# 遍历每个维度的子图
for ax, dim in zip(axes.flatten(), dimensions):
    ax2 = ax.twinx()  # 双Y轴

    # 柱状图
    width = 2
    ax.bar(cosine_values - width, data[dim]["f1_keywords"], width=2, label='Keywords', color=colors['keywords'], alpha=0.8)
    ax.bar(cosine_values, data[dim]["f1_metrics"], width=2, label='All Metrics', color=colors['metrics'], alpha=0.8)
    ax.bar(cosine_values + width, data[dim]["f1_both"], width=2, label='Both', color=colors['both'], alpha=0.8)


    # 折线图
    ax2.plot(cosine_values, data[dim]["precision_keywords"], marker='o', linestyle='-', color=colors['keywords'], alpha=0.8)
    ax2.plot(cosine_values, data[dim]["precision_metrics"], marker='o', linestyle='-', color=colors['metrics'], alpha=0.8)
    ax2.plot(cosine_values, data[dim]["precision_both"], marker='o', linestyle='-', color=colors['both'], alpha=0.8)


    # 坐标轴设置
    ax.set_xlabel('Cosine Threshold')
    ax.set_ylabel('F1 Score')
    ax2.set_ylabel('Precision Score')
    ax.set_title(f'Dimension = {dim}')
    ax.set_ylim(20, 100)
    ax2.set_ylim(20, 100)

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.5)

# 调整布局
fig.tight_layout(rect=[0, 0, 1, 0.95])

# 添加图例（只在一个子图中显示）
axes[0, 0].legend(loc='upper left')
# plt.savefig('./figures/performance_1.png', dpi=300, bbox_inches='tight')
plt.savefig('./figures/performance_1.png', dpi=300, bbox_inches='tight')

plt.show()
