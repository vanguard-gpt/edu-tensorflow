import tensorflow as tf
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리 추가

# 기존 코드와 동일
x = [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]

y = [[-1],
     [-1],
     [-1],
     [1]]

w = tf.Variable(tf.random.uniform([2, 1], -0.5, 0.5))
b = tf.Variable(tf.zeros([1]))

opt = tf.keras.optimizers.SGD(learning_rate=0.1)

def forward():
    s = tf.add(tf.matmul(x, w), b)
    o = tf.tanh(s)
    return o

def loss():
    o = forward()
    return tf.reduce_mean((y - o) ** 2)

def train_step():
    with tf.GradientTape() as tape:
        l = loss()
        grad = tape.gradient(l, [w, b])
        opt.apply_gradients(zip(grad, [w, b]))

# 손실 값을 저장할 리스트 초기화
loss_values = []

# 학습 루프 실행
for i in range(1000):
    train_step()
    current_loss = loss().numpy()
    loss_values.append(current_loss)  # 손실 값 저장
    if i % 100 == 0:
        print(f"Loss at epoch {i}: {current_loss}")

print("Final weights and bias:")
print("w =", w.numpy())
print("b =", b.numpy())

o = forward()
print("Prediction:")
print(o.numpy())

# 손실 값 시각화
plt.plot(loss_values)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()