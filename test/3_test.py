import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 기존 코드와 동일
x = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])

y = np.array([[-1],
              [-1],
              [-1],
              [1]])

w = tf.Variable(tf.random.uniform([2, 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

def forward(x_input):
    s = tf.add(tf.matmul(x_input, w), b)
    o = tf.tanh(s)
    return o

def loss():
    o = forward(x)
    return tf.reduce_mean((y - o) ** 2)

def train_step():
    with tf.GradientTape() as tape:
        l = loss()
        grad = tape.gradient(l, [w, b])
        opt.apply_gradients(zip(grad, [w, b]))

# 시각화를 위한 함수 정의
def plot_decision_boundary(epoch):
    # 격자 생성
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 모델 예측
    Z = forward(grid)
    Z = Z.numpy().reshape(xx.shape)
    # 그래프 그리기
    plt.contourf(xx, yy, Z, levels=[-1.0, 0.0, 1.0], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    plt.title(f'Decision Boundary at Epoch {epoch}')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.show()

# 학습 루프 실행
epochs = 1000
for i in range(epochs):
    train_step()
    if i % 200 == 0:
        print(f"Loss at epoch {i}: {loss().numpy()}")
        plot_decision_boundary(i)

print("Final weights and bias:")
print("w =", w.numpy())
print("b =", b.numpy())

# 최종 결정 경계 출력
plot_decision_boundary(epochs)