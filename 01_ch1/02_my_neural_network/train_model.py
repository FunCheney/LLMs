"""
训练模型
    训练模型时机器学习和深度学习中的核心过程，旨在通过大量数据学习模型参数，以便能够对新的、未见过的数据做出准确的预测
训练模型通常包括的步骤
    1. 数据准备
        * 收集和处理数据，包括清洗、标准化和归一化
        * 将数据分为训练集与测试集
    2. 定义模型
        * 选择模型架构，例如决策树、神经网络
        * 初始化模型参数（权重和偏置）
    3. 选择损失函数
        * 根据任务类型（分类、回归）选择合适的损失函数
    4. 选择优化器
        * 选择一个优化算法， 如 SGD、Adam 等
    5. 前向传播
        * 在每次迭代中，将输入数据通过模型传递，计算预测输出
    6. 计算损失哈数
        * 使用损失函数评估输出与真实标签之间的差异
    7. 反向传播
        * 利用自求导计算损失函数相当于模型参数的结果
    8. 更新参数
        * 根据计算出的梯度和优化器的策略更新模型和参数
    9. 迭代优化
        * 重复步骤5-8，直到模型在验证集上的性能不再提升或达到预定的迭代次数。
    10. 评估和测试
        * 使用测试集评估模型的最终性能，确保模型没有过拟合。
    11. 模型调优
        * 根据模型在测试集上的表现进行调参，如改变学习率、增加正则化等。
    12. 部署模型
        * 将训练好的模型部署到生产环境中，用于实际的预测任务。
"""

import torch.nn as nn
import torch.optim as optim
import torch

# 1. 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.cf1 = nn.Linear(2, 2) # 输出层到隐藏层
        self.cf2 = nn.Linear(2, 1) # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.cf1(x))
        x = self.cf2(x)
        return x

# 2. 创建模型实例
model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss() # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 优化器

# 4. 假设我们有训练数据 X 和 Y
X = torch.randn(10, 2) # 10 个样本，2个特征
Y = torch.randn(10, 1) # 10 个目标值

# 5.循环训练
for epoch in range(100): # 训练 100 轮
    optimizer.zero_grad() # 清空之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y) # 计算损失
    loss.backward() # 反向传播
    optimizer.step() # 更新参数
    # 每 10 轮输出一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')





































