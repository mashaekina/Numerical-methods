import numpy as np
import pandas as pd
from decimal import Decimal, getcontext

# Параметры выборок
samples_params = [
    (1, 1, "μ=1, σ=1"),
    (10, 0.1, "μ=10, σ=0.1"),
    (100, 0.01, "μ=100, σ=0.01")
]
n = 1000

np.random.seed(111)  # начальное состояние для генератора случайных чисел

getcontext().prec = 50 #ставим точнось 50 значащих цифр для эталона

# Эталон
def exact_variance_decimal(data):
    dec_data = [Decimal(x) for x in data]
    n = Decimal(len(data))
    mean = sum(dec_data) / n
    var = sum((x - mean) ** 2 for x in dec_data) / n
    return float(var)

#Двухпроходной метод
def two_pass(data, dtype):
    x = np.array(data, dtype=dtype)
    mean = np.mean(x)
    var = np.mean((x - mean) ** 2)
    return var

#Быстрый метод
def fast_pass(data, dtype):
    x = np.array(data, dtype=dtype)
    mean = np.mean(x)
    mean_sq = np.mean(x ** 2)
    return mean_sq - mean ** 2

#Однопроходной метод
def one_pass(data, dtype):
    n = 0
    mean = dtype(0)
    M2 = dtype(0)  #сумма квадратов отклонений от текущего среднего
    for val in data:
        x = dtype(val)
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    var = M2 / n
    return var


results = []
for mu, sigma, desc in samples_params:
    # Генерация выборки
    sample = np.random.normal(mu, sigma, n).tolist()

    ref = exact_variance_decimal(sample)

    for dtype_str, dtype in [('float32', np.float32), ('float64', np.float64)]:
        v_two = two_pass(sample, dtype)
        v_fast = fast_pass(sample, dtype)
        v_one = one_pass(sample, dtype)

        # Погрешности
        for name, val in [('2-pass', v_two), ('fast-pass', v_fast), ('1-pass', v_one)]:
            abs_err = abs(val - ref)
            rel_err = abs_err / abs(ref) if ref != 0 else np.nan
            results.append({
                'Выборка': desc,
                'Тип': dtype_str,
                'Метод': name,
                'Значение': val,
                'Абс. ошибка': abs_err,
                'Отн. ошибка': rel_err
            })

df = pd.DataFrame(results)
print(df.to_string(index=False))
