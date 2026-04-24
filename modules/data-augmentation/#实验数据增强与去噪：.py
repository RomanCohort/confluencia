#实验数据增强与去噪：
# 湿实验的预实验数据往往样本量较小。
# 利用生成对抗网络（GAN）或变分自编码器（VAE）学习现有数据的分布。
# 作用： 模拟出更多可能的实验结果
# 帮助团队在数据稀缺的情况下发现隐藏的规律
# 指导后续正式实验的样本量设计。
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Lambda
import keras.backend as K
from keras.losses import mse
import matplotlib.pyplot as plt
# 生成模拟实验数据
def generate_data(num_samples=1000):
    x = np.random.uniform(-1, 1, (num_samples, 1))
    y = 3 * x + np.random.normal(0, 0.1, (num_samples, 1))  # 线性关系加噪声
    return x, y
# 定义生成器模型
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='linear'))
    return model
# 定义判别器模型
def build_discriminator():
    model = Sequential()
    model.add(Dense(32, input_dim=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model
# 定义GAN模型
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model
# 训练GAN模型
def train_gan(generator, discriminator, gan, x_train, epochs=10000, batch_size=32):
    half_batch = batch_size // 2
    for epoch in range(epochs):
        # 训练判别器
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_samples = x_train[idx]
        noise = np.random.normal(0, 1, (half_batch, 5))
        fake_samples = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, 5))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")
# 主程序
if __name__ == "__main__":
    x, y = generate_data(1000)
    x_train, x_test = train_test_split(y, test_size=0.2, random_state=42)
    latent_dim = 5
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    gan = build_gan(generator, discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    train_gan(generator, discriminator, gan, x_train, epochs=10000, batch_size=32)
    # 生成新的数据样本
    noise = np.random.normal(0, 1, (100, latent_dim))
    generated_samples = generator.predict(noise)
    # 可视化结果
    plt.scatter(x, y, label='Original Data', alpha=0.5)
    plt.scatter(generated_samples, np.zeros_like(generated_samples), color='r', label='Generated Data', alpha=0.5)
    plt.legend()
    plt.show()
