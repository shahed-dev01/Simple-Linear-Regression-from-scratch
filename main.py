import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name': ['Shahed', 'Ratul', 'Sofi', 'Azad', 'Pobon', 'Rifat'],
    'Study_hours': [2, 1, 3, 5, 4, 7],
    'Marks': [45, 27, 59, 78, 75, 98]
        }

df = pd.DataFrame(data)

X = df.loc[:, 'Study_hours']
y = df['Marks']

m = 0
b = 0

epochs = 10000

for epoch in range(epochs):


    y_predict = (m*X)+b

    cost = y - y_predict
    mse = cost ** 2
    mse_mean = np.mean(mse)

    n = len(X)
    Dm = (-2/n) * sum(X * (y - y_predict))
    Db = (-2/n) * sum(y - y_predict)

    L = 0.01

    m = m - (L*Dm)
    b = b - (L*Db)
print('Slope:',m)
print('y intercept:',b)

plt.scatter(X, y, alpha=0.5)
plt.plot(X, y_predict, color='red', label='Best fit line')
plt.title('Simple Linear Regression')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.legend()
plt.show()
