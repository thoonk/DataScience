from scratch.statistics import correlation, mean, standard_deviation

def predict(alpha, beta, x_i): #선형모델
    return beta * x_i + alpha
def error(alpha, beta, x_i, y_i): # 모델과 실제의 에러 계산è 음수문제
    return predict(alpha, beta, x_i) - y_i


# 최소자승법: 모델과 실제의 에러제곱의 합을 최소로하는 알파, 베타 찾는 방법
def sum_of_sqerrors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
    for x_i, y_i in zip(x, y))


# 최소자승법 계수 구하기: 복잡한 algebra 를 거치면 다음 결과를 얻음
def least_squares_fit(x,y):
    """given training values for x and y, find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))
def r_squared(alpha, beta, x, y):
    """the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model"""
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
    total_sum_of_squares(y))

from matplotlib import pyplot as plt
from scratch.statistics import num_friends_good, daily_minutes_good

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

#Should find that y = 3x -5
print(least_squares_fit(x,y))

# 1. least_squares_fit
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
print("alpha", alpha)
print("beta", beta)
lst_y = [beta*x + alpha for x in range(1, 60, 1)]
plt.scatter(num_friends_good,daily_minutes_good)
plt.plot(lst_y)
plt.show()

from scratch.linear_algebra import Vector, scalar_multiply, add

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

from scratch.gradient_descent import gradient_step
import random, tqdm

num_epochs = 10000
random.seed(0)
guess = [random.random(), random.random()] # choose random value to start
learning_rate = 0.00001
with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess
        # Partial derivative of loss with respect to alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
        for x_i, y_i in zip(num_friends_good,
        daily_minutes_good))
        # Partial derivative of loss with respect to beta
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
        for x_i, y_i in zip(num_friends_good,
        daily_minutes_good))
        # Compute loss to stick in the tqdm description
        loss = sum_of_sqerrors(alpha, beta,
        num_friends_good, daily_minutes_good)
        t.set_description(f"loss: {loss:.3f}")
        # Finally, update the guess
        guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)
# We should get pretty much the same results:
alpha, beta = guess
print(alpha)
print(beta)
