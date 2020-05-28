import random
import matplotlib.pyplot as plt
from typing import List, Callable, TypeVar, Iterator
from scratch.linear_algebra import distance, add, scalar_multiply, vector_mean

Vector = List[float]
# 경사 하강법
# 그래디언트 계산하기
def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x+h) - f(x)) / h

def plot_estimated_derivative():
    def square(x):
        return x*x
    def derivative(x):
        return 2 * x

    xs = range(-10, 11)
    actuals = [derivative(x) for x in xs]
    estimate = [difference_quotient(square, x, h= 0.001) for x in xs]

    plt.title("Actual Derivatives vs Estimates")
    plt.plot(xs, actuals, 'rx', label = 'Actual')
    plt.plot(xs, estimate, 'b+', label='Estimates')
    plt.legend(loc=9)
    plt.show()

plot_estimated_derivative()

def partial_difference_quotient(f: Callable[[Vector], float], v: Vector, i: int, h: float) -> float:
    # 함수 f의 i번째 편도함수가 v에서 가지는 값
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]

def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.0001):
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

#그레디언트 적용하기
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    # v에서 step_size만큼 이동하기
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# 임의의 시작점을 선택
v = [random.uniform(-10, 10)for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v) # v의 그래디언트 계산
    v = gradient_step(v, grad, -0.01) # 그래디언트의 음수만큼 이동`
    print(epoch, v)

assert distance(v, [0,0,0]) < 0.001 # v는 0에 수렴해야 한다.

#경사 하강법으로 모델 학습
# 한 개의 데이터 포인트에서 오차의 그래디언트를 계산해 주는 함수
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept # 모델의 예측값
    error = (predicted - y)  # 오차는 (예측값 - 실제값)
    squared_error = error ** 2 # 오차의 제곱을 최소화하자
    grad = [2 * error * x, 2 * error] # 그래디언트를 사용한다.
    return grad

# 전체 데이터 셋에서 평균 제곱 오차의 그래디어튼를 계산해 주는 함수
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

for epoch in range(5000):
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    theta = gradient_step(theta, grad, -learning_rate)

slope, intercept = theta
assert 19.9 < slope < 20.1
assert 4.9 < intercept < 5.1

#미니배치 경사 하강법
T = TypeVar('T') # 변수의 타입과 무관한 함수를 생성

def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    #dataset에서 batch_size만큼 데이터 포인트를 샘플링해서 미니배치를 생성
    #각 미니배치의 시작점인 0, batch_size, 2 * batch_size, ...을 나열
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts) #미니배치의 순서를 섞는다.

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

    #미니배치
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1
    assert 4.9 < intercept < 5.1

    #SGD(stochastic gradient descent)
    for epoch in range(500):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1
    assert 4.9 < intercept < 5.1