import pandas as pd
import os
print("当前工作目录为：", os.getcwd())


# 1. 读取数据
# 假设数据是用制表符或空格分隔的
data = pd.read_csv('final_project/seeds_dataset.txt', delim_whitespace=True, header=None)

# 2. 设置列名
columns = [
    "area", "perimeter", "compactness", "length_of_kernel",
    "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class_label"
]
data.columns = columns

# 3. 查看数据基本信息
print("数据前几行：")
print(data.head())

print("\n数据基本信息：")
print(data.info())

# 4. 处理缺失值（如果有空值）
# 用平均值填充缺失值
data.fillna(data.mean(), inplace=True)

# 5. 描述性统计
print("\n描述性统计：")
print(data.describe())

# 6. 保存为CSV格式
data.to_csv('processed_data.csv', index=False, encoding='utf-8')
print("\n数据已保存为processed_data.csv")

# 7. 可视化数据（可选）
import matplotlib.pyplot as plt
import seaborn as sns

# 绘制每个特征的直方图
data.hist(bins=20, figsize=(15, 10))
plt.suptitle("Feature Distributions")
plt.show()

# 绘制特征相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
