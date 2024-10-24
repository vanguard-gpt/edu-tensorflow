import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as ds

(x_train, y_train), (x_test, y_test) = ds.mnist.load_data()
print("MNIST: ", x_train.shape, y_train.shape)
print("Test: ", x_test.shape, y_test.shape)

# fig, axs = plt.subplots(1, 10, figsize=(10, 10)) #subplots 함수를 사용하여 1행 10열로 구성된 여러 개의 서브플롯을 생성
# for i in range(10) :
#     axs[i].imshow(x_train[i], cmap='gray_r')
#     axs[i].axis('off')
    
# plt.show()

(a_train, b_train), (a_test, b_test) = ds.fashion_mnist.load_data()
print("MNIST: ", a_train.shape, b_train.shape)
print("Test: ", a_test.shape, b_test.shape)

# text_labels = ['티셔츠', '바지', '풀오버', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '부츠']

# print(b_train[:10])

# fig, axs = plt.subplots(1, 10, figsize=(10, 10)) #빈칸만 만들기
# for i in range(10) :
#     axs[i].imshow(a_train[i], cmap='gray_r')
#     axs[i].axis('off')

# plt.show()

(c_train, d_train), (c_test, d_test) = ds.cifar10.load_data()
print("CIFAR10: ", c_train.shape, d_train.shape)

# one - hot - encoding
dd_train = tf.one_hot(d_train, 10 , dtype=tf.int8)

# 원핫인코딩 : 기존 범주형 데이터(클래스 등) 를 ai 가 우열관계에 있는 수 데이터로 인식하지 않게끔, 이를 이진 벡터로 변환하여 표현하는것.
# 이렇게하면 모든 범주간의 동일한 거리를 유지할 수 있게끔 해줌.

print("정답의 one-hot : ", dd_train[:10])