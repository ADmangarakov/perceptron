import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Multiplication:
    def __init__(self):
        self.mul_l = None
        self.mul_r = None

    def forward(self, mul_l, mul_r):
        self.mul_l = mul_l
        self.mul_r = mul_r
        return np.matmul(mul_l, mul_r)

    def left_backward(self, grad):
        return np.matmul(grad, np.asmatrix(self.mul_r).transpose())

    def right_backward(self, grad):
        return np.matmul(np.asmatrix(self.mul_l).transpose(), grad)


class Sigmoid:
    def __init__(self):
        self.input = None
        self.sig_map = np.vectorize(self.sig)
        self.grad_sig_map = np.vectorize(self.grad_sig)

    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def grad_sig(self, x):
        return self.sig(x) * (1 - self.sig(x))

    def forward(self, input_frame):
        self.input = input_frame
        return self.sig_map(input_frame)

    def backward(self, grad):
        return np.multiply(np.asmatrix(self.grad_sig_map(self.input.copy())), grad)


class Layer:
    def __init__(self, row, col, learning_rate):
        self.weights = np.random.normal(0.0, 2 ** -0.5, (row, col))
        self.learning_rate = learning_rate

    def learn(self, grad):
        self.weights = np.add(self.weights, grad * self.learning_rate)


class NN:
    def __init__(self):
        # self.mse_layer = MSE(correct_prediction)
        self.w1 = Layer(16, 60, learning_rate=0.001)
        self.w2 = Layer(60, 47, learning_rate=0.001)
        self.w3 = Layer(47, 2, learning_rate=0.001)
        self.sig1 = Sigmoid()
        self.sig2 = Sigmoid()
        self.sig3 = Sigmoid()
        self.mul1 = Multiplication()
        self.mul2 = Multiplication()
        self.mul3 = Multiplication()
        self.cur_frame = None

    def forward(self, cur_frame):
        self.cur_frame = cur_frame.replace(np.nan, 0)
        f = self.mul1.forward(self.cur_frame, self.w1.weights)
        f = self.sig1.forward(f)
        f = self.mul2.forward(f, self.w2.weights)
        f = self.sig2.forward(f)
        f = self.mul3.forward(f, self.w3.weights)
        f = self.sig3.forward(f)
        return f

    def backward(self, error):
        g = error
        g = self.sig3.backward(g)
        g_w3 = self.mul3.right_backward(g)
        self.w3.learn(g_w3)
        g = self.mul3.left_backward(g)
        g = self.sig2.backward(g)
        g_w2 = self.mul2.right_backward(g)
        self.w2.learn(g_w2)
        g = self.mul2.left_backward(g)
        g = self.sig1.backward(g)
        g_w1 = self.mul1.right_backward(g)
        g_w1 = np.multiply(g_w1, np.asmatrix(self.cur_frame.notna() * 1).transpose())
        self.w1.learn(g_w1)


table = pd.read_csv('prepared_data.csv', encoding='UTF-8',
                    delimiter=',', header=[0])


def norm(x, min_v, max_v):
    return (x - min_v) / (max_v - min_v)


for column_name in table.columns:
    table[column_name] = pd.to_numeric(table[column_name], errors='coerce')

for i, row in table.iterrows():
    for column in table.columns:
        min_v = table[column].min()
        max_v = table[column].max()
        table.at[i, column] = norm(table.at[i, column], min_v=min_v, max_v=max_v)

print(table)

correct_prediction = table[['G_total', 'КГФ']]
data = table.drop(['G_total', 'КГФ'], axis=1)

epochs = 10000
lr = 0.001
nn = NN()
start_index = 0
loss_test = []
loss_validation = []
count = 0
prev_loss_valid = 1
for e in range(epochs):
    E = []
    for i in range(74):
        frame = data.iloc[i]
        predicted = nn.forward(frame)
        error_KGF = None
        error_Gtotal = None
        cK = correct_prediction.loc[frame.name].loc['КГФ']
        if not np.isnan(cK):
            error_KGF = cK - np.asmatrix(predicted).item(0, 0)

        cG = correct_prediction.loc[frame.name].loc['G_total']
        if not np.isnan(cG):
            error_Gtotal = cG - np.asmatrix(predicted).item(0, 1)
        if error_KGF is None:
            error_KGF = 0
        if error_Gtotal is None:
            error_Gtotal = 0
        E.append((error_KGF ** 2 + error_Gtotal ** 2) / 2)
        nn.backward(np.matrix([[error_KGF, error_Gtotal]]))

    loss_test.append(np.mean((np.array(E))))

    E = []
    for i in range(18):
        i += 74
        frame = data.iloc[i]
        predicted = nn.forward(frame)
        error_KGF = None
        error_Gtotal = None
        cK = correct_prediction.loc[frame.name].loc['КГФ']
        if not np.isnan(cK):
            error_KGF = cK - np.asmatrix(predicted).item(0, 0)

        cG = correct_prediction.loc[frame.name].loc['G_total']
        if not np.isnan(cG):
            error_Gtotal = cG - np.asmatrix(predicted).item(0, 1)
        if error_KGF is None:
            error_KGF = 0
        if error_Gtotal is None:
            error_Gtotal = 0
        E.append((error_KGF ** 2 + error_Gtotal ** 2) / 2)
    cur_loss_validation = np.mean(np.array(E))
    loss_validation.append(cur_loss_validation)
    if prev_loss_valid < cur_loss_validation:
        count += 1
        if count > 10:
            epochs = e
            break
    else:
        count = 0
    prev_loss_valid = cur_loss_validation

if count != 0:
    epochs = epochs - count + 1
x = np.arange(0, epochs)
loss_test = loss_test[:-count]
loss_validation = loss_validation[:-count]
plt.plot(x, loss_test)
plt.plot(x, loss_validation)
plt.legend(["test", "validation"])
plt.savefig('loss.png')
plt.show()
