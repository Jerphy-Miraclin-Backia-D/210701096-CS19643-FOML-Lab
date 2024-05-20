import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df_original = pd.read_csv("Dhaka Rent.csv", sep=";")
df = pd.read_csv("Dhaka Rent.csv", sep=";")

plt.scatter(df_original['area'], df_original['rent'], s=5)
plt.show()

#min-max normalization
df.area=(df_original.area - df_original.area.min())/(df_original.area.max()-df_original.area.min())
df.rent=(df_original.rent - df_original.rent.min())/(df_original.rent.max()-df_original.rent.min())
df.head()

plt.scatter(df['area'], df['rent'], s=5)
plt.show()

train = df.iloc[:40] #remaining 20
test = df.iloc[40: 60]
print(train.shape, test.shape)

reg = LinearRegression()
reg = reg.fit(train[['area']], train.rent)

plt.plot(test.area, reg.predict(test[['area']]))
plt.scatter(test.area, test.rent, s=5, c='red')
plt.show()

offset = (df_original.rent.max() - df_original.rent.min()) + df_original.rent.min()
pred = reg.predict([[test.area.iloc[0]]])[0]
print("Original value ", test.rent.iloc[0] * offset)
print("Predicted value ", pred * offset)
print("Error ", round(abs(test.rent.iloc[0] - pred) * 100 / test.rent.iloc[0], 2), "%")
