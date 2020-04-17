from typing import List
import math

Vector = List[float]

def dot(v:Vector, w:Vector) -> float:
    if len(v) == len(w):
        return sum(v_i * w_i for v_i, w_i in zip(v,w))
    else:
        print("")

def sum_of_squares(v: Vector) -> float:
    return dot(v,v)

def mean(v:List[float]) -> float:
    return (sum(v) / len(v))

def de_mean(xs:List[float]):
    xs_bar = mean(xs)
    return [x - xs_bar for x in xs]


def variance(xs:List[float]) -> float:
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n -1)

def standard_deviations(xs: List[float]) -> float:
    return math.sqrt(variance(xs))

#공분산
def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "두 리스트의 길이가 같아야 합니다."
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

#상관계수
def correlation(xs: List[float], ys: [float]) -> float:
    stdev_x = standard_deviations(xs)
    stdev_y = standard_deviations(ys)
    if stdev_x >0 and stdev_y >0 :
        return covariance(xs,ys) /stdev_x / stdev_y
    else:
        return 0

#실습
#상관계수가 -1인 경우
num_friends_neg1 = [20, 25, 30, 35, 40]
daily_minutes_neg1 = [60, 55, 50, 45, 40]
print(correlation(num_friends_neg1, daily_minutes_neg1))
#상관계수가 0인 경우
num_friends_0 = [20, 20, 20, 20, 20]
daily_minutes_0 = [40, 40, 40, 40, 40]
print(correlation(num_friends_0, daily_minutes_0))
#상관계수가 1인 경우
num_friends_1 = [20,25,30,35,40]
daily_minutes_1 = [40,45,50,55,60]
print(correlation(num_friends_1, daily_minutes_1))

