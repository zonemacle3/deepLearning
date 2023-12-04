# tensorflow 1.0 version
# import tensorflow as tf
# import numpy as np

# w = tf.Variable(tf.random.normal([1]), name = 'weight')
# b = tf.Variable(tf.random.normal([1]), name = 'bias')
# x = tf.TensorSpec(shape=[None], dtype=tf.float32)
# y = tf.TensorSpec(shape=[None], dtype=tf.float32)

# hypothesis = x*w+b

# cost = tf.reduce_mean(tf.square(hypothesis-y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2001):
#     cost_val, w_val, b_val, _ = sess.run([cost, w, b, train],
#          feed_dict = {x:[1,2,3,4,5],y:[2.1,3.1,4.1,5.1,6.1]})
#     if step % 20 == 0:
#         print(step, cost_val, w_val, b_val)

# tensorflow 2.0 version 
import tensorflow as tf
import numpy as np

# 변수 초기화
w = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# 입력 데이터와 출력 데이터 생성 
x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
y = tf.constant([2.1, 3.1, 4.1, 5.1, 6.1], dtype=tf.float32)

# 가설 생성
hypothesis = x * w + b

# 손실 함수 정의
cost = tf.reduce_mean(tf.square(hypothesis - y))
# tr.reduce_mean = 평균값을 계산 , hypothesis -y = 가설과 실제 값의 차이 

# 옵티마이저 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#  tf.keras.optimizers.Adam = Adam 옵티마이저를 생성 
#   w = w - learning_rate * gradients[w]
#   b = b - learning_rate * gradients[b]

# 학습 루프 
def train_step(x, y):
  with tf.GradientTape() as tape: # 그래프 테이프를 사용해서 손실 함수의 미분을 계산한다.
    hypothesis = x * w + b # 가설을 계산
    loss = tf.reduce_mean(tf.square(hypothesis - y)) #손실 함수를 계산
  gradients = tape.gradient(loss, [w, b]) # 손실 함수의 미분을 계산한다.
  optimizer.apply_gradients(zip(gradients, [w, b])) # 옵티마이저를 사용해서 변수를 업데이트한다. 
  # zip(gradients, [w,b])는 가중치와 편향의 값에 대한 미분과 변수들을 묶은 튜플을 생성 

for step in range(2001):
  train_step(x, y) # 함수 호출은 각 학습 과정에서 가주치와 편향을 업데이트 한다 
  if step % 20 == 0: # 조건문은 20번마다 손실 함수의 값, 가중치의 값, 편향의 값을 출력 
    print(step, cost.numpy(), w.numpy(), b.numpy())

print(w.numpy(), b.numpy()) # 최종 가중치와 편향의 값을 출력 
