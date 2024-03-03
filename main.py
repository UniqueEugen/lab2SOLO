import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Ваши данные
X = np.array([120, 150, 140, 156, 216, 156, 153, 324, 102, 233, 200, 123])
Y = np.array([354, 325, 205, 385, 524, 562, 655, 244, 241, 145, 254, 145])
#X = np.array([100, 90, 150, 31, 60, 39, 40, 70, 80, 150, 120, 130])
#Y = np.array([131, 110, 170, 141, 150, 160, 200, 230, 240, 260, 270, 300])
# Расчет средних значений
X_mean = np.mean(X)
Y_mean = np.mean(Y)
print("Cреднее X = ", X_mean, " Среднее Y = ", Y_mean)

# Вычисление разностей
delta_X = X - X_mean
delta_Y = Y - Y_mean

# Вычисление произведений
delta_XY = delta_X * delta_Y
#print("Дельта XY = ", delta_XY)

# Вычисление квадратов delta_X
delta_X_squared = delta_X ** 2
#print("Дельта X квадрат = ", delta_X_squared)

# Оценки коэффициентов уравнения регрессии
b1 = np.sum(delta_XY) / np.sum(delta_X_squared)
b0 = Y_mean - b1 * X_mean

# Уравнение регрессии
regression_equation = f"Ŷ = {b0:.2f} + {b1:.2f} * X"

# Добавление константы к X
X_with_const = sm.add_constant(X)

# Построение модели линейной регрессии
model = sm.OLS(Y, X_with_const)
results = model.fit()

# Построение линии тренда
X_line = np.linspace(X.min(), X.max(), 100)
Y_line = b0 + b1 * X_line

plt.scatter(X, Y)
plt.plot(X_line, Y_line, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Корелляционное поле и линия треда')
plt.grid(True)
plt.show()

# Вывод результатов
print("Уравнение регрессии:", regression_equation)
print("Статистический анализ:")
print(results.summary())

# Значение линейного коэффициента парной корреляции
correlation_coefficient = np.sqrt(results.rsquared)

# Стандартная ошибка остаточной компоненты
residual_std_error = np.sqrt(results.mse_resid)

# Стандартные ошибки оценивания коэффициента b и свободного члена а
b_std_error, a_std_error = np.sqrt(np.diag(results.cov_params()))

# t-критерий Стьюдента для обоих параметров
t_values = results.tvalues

# Вывод результатов
print("Значение линейного коэффициента парной корреляции:", correlation_coefficient)
print("Стандартная ошибка остаточной компоненты:", residual_std_error)
print("Стандартная ошибка оценивания коэффициента b:", b_std_error)
print("Стандартная ошибка оценивания свободного члена а:", a_std_error)
print("t-критерий Стьюдента для коэффициента b:", t_values[1])
print("t-критерий Стьюдента для свободного члена а:", t_values[0])

# Расчет доверительных интервалов
conf_int_b0 = results.conf_int(alpha=0.05)[0]
conf_int_b1 = results.conf_int(alpha=0.05)[1]

# Вывод результатов
print("Доверительный интервал для a:", conf_int_b0)
print("Доверительный интервал для b:", conf_int_b1)

