import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

# plt.scatter(data.lotsize, data.price)
# plt.show()

def loss_function(m, b, points):
    totalError = 0
    for i in range(len(points)):
        x = points.iloc[i].lotsize
        y = points.iloc[i].price
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradient_decent(m_now, b_now, points, learningRate):
    m_gradient = 0
    b_gradient = 0
    N = (len(points))
    for i in range(N):
        x = float(points.iloc[i].lotsize)
        y = float(points.iloc[i].price)
        m_gradient += -(2/N) * x * (y - ((m_now * x) + b_now))
        b_gradient += -(2/N) * (y - ((m_now * x) + b_now))
    new_m = m_now - float(learningRate) * m_gradient
    new_b = b_now - float(learningRate) * b_gradient
    return [new_m, new_b]


m = 0
b = 0
learningRate = 0.000000001
num_iterations = 300

for i in range(num_iterations):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_decent(m, b, data, learningRate)

print(m, b)

plt.scatter(data.lotsize, data.price, color="black")
plt.plot(data.lotsize, m * data.lotsize + b, color="red")
plt.show()







