# Lab 3 Minimizing Cost
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# unit = 출력뉴런의 개수 , input_dim = 입력 차원 개수 
# == 이는 입력 데이터가 1차원 배열이고 출력도 1개의 값이라는 것을 의미

sgd = tf.keras.optimizers.SGD(lr=0.1) # sgd(경사 하강법) 옵티마이저를 설정 
tf.model.compile(loss='mse', optimizer=sgd) # 손실 함수로 mse(평균 제곱 오차) 사용 
# 옵티마이저로 sgd변수를 지, 손실 함수를 최소화하도록 옵티마이저를 사용해 가중치를 업데이트

tf.model.summary() # 모델의 구조와 설정을 출력 

# fit() trains the model and returns history of train
history = tf.model.fit(x_train, y_train, epochs=100)

y_predict = tf.model.predict(np.array([5, 4])) # 새로운 데이터에 대한 예측을 수행 
print(y_predict)

# Plot training & validation loss values
plt.plot(history.history['loss']) # plot 메소드를 사용해 loss의 값을 시각화함 
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()