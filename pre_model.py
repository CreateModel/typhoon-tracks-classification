import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# 生成一些示例数据，实际使用时替换为真实的台风路径数据
# 假设数据有100个样本，每个样本有50个特征
np.random.seed(42)
data = np.random.rand(100, 50)

def pre_data(data):
    # 定义自编码器的输入层
    input_layer = Input(shape=(data.shape[1],))

    # 定义编码器部分
    # 这里我们将数据降维到10维
    encoded = Dense(200, activation='relu')(input_layer)
    encoded = Dense(100, activation='relu')(encoded)

    # 定义解码器部分
    decoded = Dense(200, activation='relu')(encoded)
    decoded = Dense(data.shape[1], activation='linear')(decoded)

    # 定义自编码器模型
    autoencoder = Model(input_layer, decoded)

    # 定义编码器模型，用于提取关键特征
    encoder = Model(input_layer, encoded)

    # 编译自编码器模型
    autoencoder.compile(optimizer='adam', loss='mse')

    # 训练自编码器
    history = autoencoder.fit(data, data,
                              epochs=50,
                              batch_size=32,
                              shuffle=True)

    # 提取关键特征
    encoded_data = encoder.predict(data)

    # 绘制训练损失曲线
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    print("Encoded data shape:", encoded_data.shape)
    return encoded_data