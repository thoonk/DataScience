import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # generic type to represent a data point
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:] # 얕은 복사본을 만든다.
    random.shuffle(data) # shuffle이 리스트 내용을 바꾸기 때문
    cut = int(len(data) * prob) # prob을 사용하여 자를 위치를 선택
    return data[:cut], data[cut:] # 섞인 리스트를 자른다.

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# 자른 비율이 맞는지 확인
assert len(train) == 750
assert len(test) == 250

# 기존 데이터 유지되는지 확인
assert sorted(train + test) == data



# 독립(x), 종속(y) 형태의 데이터 분할
# 데이터(xi, yi가 페어로 셔플되도록 인덱스를 활용한 셔플과 분할
Y = TypeVar('Y')  # 출력 변수를 표현하기 위한 일반적인 타입

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # 인덱스를 생성하여 분할
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

xs = [x for x in range(20)]  # xs는 1 ... 1000
ys = [2 * x for x in xs]       # 각각 y_i 는 x_i의 두 배
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

print(xs, ys)
print(x_train, x_test)
print(y_train, y_test)

