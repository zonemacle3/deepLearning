import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential() # 입출력이 선형으로 연결 
# units == output shape, input_dim == input shape
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1)) # fully-connected 계층 , unit = 출력뉴런의 개수 

sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2
# sgd 알고리즘으로 최적화 알고리즘 객체 생성 (신경망 훈련)
# mse(평균제곱오차): 예측된 출력과 실제 출력사이 평균제곱차이를 측정 

# prints summary of the model to the terminal
tf.model.summary() # 모델 구조에 대한 요약 출력 

# fit() executes training
tf.model.fit(x_train, y_train, epochs=200) # 훈련 데이터 세트에서 모델을 훈련(x=입,y=출,epochs = 반복)

# predict() returns predicted value # 새로운 데이터에 대한 예측 출력 
y_predict = tf.model.predict(np.array([5, 4])) # 새로운 입력 데이터 포인트
# 여기서 [5,4]로 설정한 이유는 x,y값이 -3~+4 이기에 10 이하의 수를 
# 입력에 사용하는 데이터 포인트로 사용해서 출력하도록 설정한것 다른 의미있는 수는 아님 
print(y_predict) # 결과 출력 