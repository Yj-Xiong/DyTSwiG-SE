import matplotlib.pyplot as plt
import numpy as np
import os

# 数据
models = [
    "Ours-U",
    "Ours",
    "PrimeK-Net",
    "SE-Mamba",
    "MN-Net",
    "MP-SENet",
    "Ours-M",
    "Mamba-SEUNet (L)",
    "M-CMGAN",
    "CMGAN",
]
params = [0.78, 1.20, 1.41, 2.26, 1.39, 2.26, 1.54, 6.28, 1.76, 1.83]
rtfs = [0.042/2, 0.028/2, 0.039/2, 0.037/2, 0.061/2, 0.035/2, 0.044/2, 0.066/2, 0.036/2, 0.047/2]
memory_usagiess = [178.87, 182.84, 183.69, 272.98, 378.48, 669.80, 262.08, 231.51, 582.47, 597.69]
colors = ["tab:blue", "tab:green", "tab:olive","tab:orange", "tab:cyan", "tab:pink",  "black", "tab:red", "tab:purple", "tab:brown"]

# 设置图表尺寸
plt.figure(figsize=(8, 4))

# 找到 "Ours" 的索引
ours_index = models.index("Ours")
ours_index1 = models.index("Ours-U")
ours_mamba_index = models.index("Ours-M")
# 绘制散点图，点的大小随参数变化
scatter_size = [2.25 * 100 for param in params]  # 参数值的 1000 倍作为点的大小
plt.scatter(
    params,
    rtfs,
    s=scatter_size,  # 点的大小根据参数设置
    color=colors,
    marker="o",  # 使用圆形标记
    label=models,
)
# 添加红色五角星标记到 "Ours"
plt.scatter(
    params[ours_index],
    rtfs[ours_index],
    s=scatter_size[ours_index],  # 点的大小随参数变化，稍大一点
    color="red",
    marker="*",  # 设置为五角星
    label="Ours (Highlighted)"  # 添加到图例中
)
plt.scatter(
    params[ours_index1],
    rtfs[ours_index1],
    s=scatter_size[ours_index1],  # 点的大小随参数变化，稍大一点
    color="red",
    marker="*",  # 设置为五角星
    label="Ours-U"  # 添加到图例中
)
plt.scatter(
    params[ours_mamba_index],
    rtfs[ours_mamba_index],
    s=scatter_size[ours_mamba_index],  # 点的大小随参数变化，稍大一点
    color="red",
    marker="*",  # 设置为五角星
    label="Ours-M"  # 添加到图例中
)
# 添加模型名称的注释（显示在点正下方）
for i in range(len(models)):
    # 排除 "SE-Mamba" 和 "Mamba-SEUNet (L)"
    if models[i] not in ["MP-SENet", "Mamba-SEUNet (L)", "M-CMGAN", "PrimeK-Net","Ours" ]:
        plt.text(
            params[i],  # 横坐标
            rtfs[i] + 0.004,  # 纵坐标，稍向下偏移
            models[i],
            fontsize=10,
            va="top",  # 垂直对齐为顶部
            ha="center",  # 水平对齐为居中
            color=colors[i],  # 字体颜色
        )

# 为 "SE-Mamba" 和 "Mamba-SEUNet (S)" 添加模型名称的注释
# 找到它们的索引
se_mamba_index = models.index("MP-SENet")
mamba_seunet_l_index = models.index("Mamba-SEUNet (L)")
m_cmgan_index = models.index("M-CMGAN")
primek_net_index = models.index("PrimeK-Net")
# 为 "SE-Mamba" 添加注释
plt.text(
    params[se_mamba_index],  # 横坐标
    rtfs[se_mamba_index] - 0.002,  # 纵坐标
    models[se_mamba_index],  # 模型名称
    fontsize=10,
    va="top",  # 垂直对齐为顶部
    ha="center",  # 水平对齐为居中
    color=colors[se_mamba_index],  # 字体颜色
)

# 为 "Mamba-SEUNet (L)" 添加注释
plt.text(
    params[mamba_seunet_l_index],  # 横坐标
    rtfs[mamba_seunet_l_index] + 0.004,  # 纵坐标
    models[mamba_seunet_l_index],  # 模型名称
    fontsize=10,
    va="top",  # 垂直对齐为顶部
    ha="center",  # 水平对齐为居中
    color=colors[mamba_seunet_l_index],  # 字体颜色
)
plt.text(
    params[m_cmgan_index],  # 横坐标
    rtfs[m_cmgan_index] + 0.003,  # 纵坐标
    models[m_cmgan_index],  # 模型名称
    fontsize=10,
    va="top",  # 垂直对齐为顶部
    ha="center",  # 水平对齐为居中
    color=colors[m_cmgan_index],  # 字体颜色
)
# 为 "Ours(Mamba)" 添加注释
# plt.text(
#     params[ours_mamba_index],  # 横坐标
#     rtfs[ours_mamba_index] - 0.002,  # 纵坐标
#     models[ours_mamba_index],  # 模型名称
#     fontsize=10,
#     va="top",  # 垂直对齐为顶部
#     ha="center",  # 水平对齐为居中
#     color=colors[ours_mamba_index],  # 字体颜色
# )
# 为 "Ours" 添加注释
plt.text(
    params[ours_index],  # 横坐标
    rtfs[ours_index] - 0.002,  # 纵坐标
    models[ours_index],  # 模型名称
    fontsize=10,
    va="top",  # 垂直对齐为顶部
    ha="center",  # 水平对齐为居中
    color=colors[ours_index],  # 字体颜色
)
plt.text(
    params[primek_net_index],  # 横坐标
    rtfs[primek_net_index] - 0.002,  # 纵坐标
    models[primek_net_index],  # 模型名称
    fontsize=10,
    va="top",  # 垂直对齐为顶部
    ha="center",  # 水平对齐为居中
    color=colors[primek_net_index],  # 字体颜色
)

# 设置轴标签
plt.xlabel("Parameters (M)", fontsize=12)
plt.ylabel("Real-Time Factor (RTF)", fontsize=12)

# 去掉右框线和上框线
ax = plt.gca()  # 获取当前轴
ax.spines['right'].set_visible(False)  # 去掉右框线
ax.spines['top'].set_visible(False)  # 去掉上框线
ax.tick_params(direction='in', length=6, width=1.5, colors='k')

ax.spines['left'].set_linewidth(1.5)  # 加粗轴线
ax.spines['bottom'].set_linewidth(1.5)  # 加粗轴线

# 设置轴标签
plt.xlabel("Parameters (M)", fontsize=12)
plt.ylabel("Real-Time Factor (RTF)", fontsize=12)

# 设置 y 轴范围，从 0.00 开始
plt.ylim(0.00, max(rtfs) + 0.02)  # 确保 y 轴从 0.00 开始，并且适当增加上限

# 自定义图例
sorted_indices = np.argsort(memory_usagiess)[::-1]  # 从大到小排序
sorted_models = [models[i] for i in sorted_indices]
sorted_memory = [memory_usagiess[i] for i in sorted_indices]
sorted_colors = [colors[i] for i in sorted_indices]
sorted_sizes = [memory_usagiess[i] * 30 for i in sorted_indices]

# 创建空的代理对象，用于生成图例中的圆圈
legend_handles = []
for model, size, color in zip(sorted_models, sorted_sizes, sorted_colors):
    if model == "Ours" or model == "Ours-U" or model == "Ours-M":
        # 先画圆形（保持原样式）
        circle = plt.Line2D([0], [0],
                            marker='|',
                            color=color,
                            markersize=np.sqrt(size) / 10,
                            markeredgewidth=10,
                            linestyle='None')
        # 再叠加红色五角星（居中，稍小）
        star = plt.Line2D([0], [0],
                          marker='*',
                          color='red',
                          markersize=np.sqrt(size) / 10,  # 比圆圈小一些
                          linestyle='None')  # 可选边框
        legend_handles.append((circle, star))  # 将两个句柄作为元组传递
    else:
        # 其他模型正常画圆形
        handle = plt.Line2D([0], [0],
                           marker='|',
                           color=color,
                           markersize=np.sqrt(size) / 10,
                           markeredgewidth=10,
                           linestyle='None')
        legend_handles.append(handle)

# 手动构建图例的句柄和标签顺序
custom_handles = [
    legend_handles[0],
    legend_handles[3],
    legend_handles[6],
    legend_handles[1],
    legend_handles[4],
    legend_handles[7],
    legend_handles[2],
    legend_handles[5],
    legend_handles[8],
]

# 自定义图例标签
custom_labels = [
    sorted_memory[0],
    sorted_memory[3],
    sorted_memory[6],
    sorted_memory[1],
    sorted_memory[4],
    sorted_memory[7],
    sorted_memory[2],
    sorted_memory[5],
    sorted_memory[8],
]

# 展示图例
plt.legend(
    handles=legend_handles,
    labels=sorted_memory,
    title="Max Memory Usage (MB) of Processing 2-second Audio",
    loc='upper right',
    bbox_to_anchor=(1., 1.0),  # 将图例放在右侧外侧
    ncol=len(sorted_models) // 2,  # 关键修改：设置列数为图例项的总数，实现横向排列
    fontsize=10,
    frameon=True,
    facecolor='white',
    handletextpad=0.5,  # 调整文本与句柄的间距
    borderaxespad=0.5,
    labelspacing=0.5,
)

# 调整 y 轴范围，确保名称不会被截断
# plt.ylim(-0.005, max(rtfs) + 0.02)  # 适当增加 y 轴上限（已移除）
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.0, hspace=0.0)
plt.margins(x=0.1, y=0.1)  # 增大 x 和 y 方向的边距
# 显示网格
plt.grid(True, linestyle="--", alpha=0.6)
# 显示图表
# plt.subplots_adjust(bottom=0.2)  # 增加底部空间
plt.tight_layout()  # 自动调整布局
plt.show()
