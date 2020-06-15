import pandas as pd
import random
import tqdm
from typing import TypeVar, List, Tuple
from scratch.linear_algebra import dot, vector_mean
from scratch.gradient_descent import gradient_step
from scratch.simple_linear_regression import total_sum_of_squares

Vector = List[float]
X = TypeVar('X')


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]  # 얕은 복사본을 만든다.
    # random.shuffle(data) # shuffle이 리스트 내용을 바꾸기 때문
    cut = int(len(data) * prob)  # prob을 사용하여 자를 위치를 선택
    return data[:cut], data[cut:]  # 섞인 리스트를 자른다.


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector):
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


def squared_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    # 미적분 이용한 gradient 계산, squared_error 최소화
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
    guess = [random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]

            gradient = vector_mean([squared_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_error = sum(error(x, y, beta) ** 2
                               for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_error / total_sum_of_squares(ys)


def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('data2.csv', sep=',', engine='python')
    train, test = split_data(df, 0.7)

    train_x = train[['fdust', 'nd', 'cm']].values.tolist()
    train_y = train['ufdust'].values.tolist()

    test_x = test[['fdust', 'nd', 'cm']].values.tolist()
    test_y = test['ufdust'].values.tolist()


    beta = least_squares_fit(train_x, train_y)

    r = multiple_r_squared(test_x, test_y, beta)

    print("beta: " + str(beta))
    print("r^2: " + str(r))


if __name__ == '__main__':
    main()
