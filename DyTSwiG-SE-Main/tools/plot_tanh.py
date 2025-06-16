import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-6, 6, 1000)
y1 = np.tanh(4 * x)  # tanh(4x)
y2 = np.tanh(x)      # tanh(x)
y3 = np.tanh(x / 4)  # tanh(x/4)

# 绘制曲线
plt.figure(figsize=(8, 5))
plt.plot(x, y1, linestyle='--', color='purple', label=r'$\tanh(4x)$')
plt.plot(x, y2, linestyle='-.', color='orange', label=r'$\tanh(x)$')
plt.plot(x, y3, linestyle=':', color='blue', label=r'$\tanh(x/4)$')

# 设置坐标轴
plt.xlim(-6, 6)
plt.ylim(-1.1, 1.1)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend(loc='upper left', fontsize=10)

# 添加标题和说明
plt.title(r'$\tanh(\alpha x)$ with three different $\alpha$ values.', fontsize=12)
plt.xlabel('x', fontsize=10)
plt.ylabel(r'$\tanh(\alpha x)$', fontsize=10)

# 调整背景和边框
# plt.gca().set_facecolor('#f2f2f2')  # 灰色背景
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图表
plt.tight_layout()
plt.show()
