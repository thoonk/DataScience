from typing import TypeVar, List, Tuple
from collections import Counter
from scratch.linear_algebra import distance
import pandas as pd

X = TypeVar('X')  # generic type to represent a data point

Vector = List[float]
airquality_lst = ['best', 'better', 'good', 'normal', 'bad', 'worse', 'serious', 'worst']


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]  # 얕은 복사본을 만든다.
    # random.shuffle(data) # shuffle이 리스트 내용을 바꾸기 때문
    cut = int(len(data) * prob)  # prob을 사용하여 자를 위치를 선택
    return data[:cut], data[cut:]  # 섞인 리스트를 자른다.


def majority_vote(labels):
    """labels는 가장 가까운 데이터부터 가장 먼 데이터 순서로 정렬되어 있다고 가정"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner  # 1등이 하나이기 때문에 반환
    else:
        return majority_vote(labels[:-1])  # 가장 먼 데이터를 제외하고 다시 찾아 본다.


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # 레이블된 포인트를 가장 가까운 데이터부터 가장 먼 데이터 순서로 정렬
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[:5], new_point[:5]))

    # 가장 가까운 k 데이터 포인트의 레이블을 살펴보고
    k_nearest_labels = [label[5] for label in by_distance[:k]]

    # 투표한다.
    return majority_vote(k_nearest_labels)


def main():
    df = pd.read_csv('data2.csv', sep=',', engine='python')

    for i in df.columns[:5]:
        del df[i]

    train, test = split_data(df, 0.7)
    test = test.reset_index() # 인덱스 0부터 시작하게 만듬
    del test['index']
    #print(test)
    # 자른 비율이 맞는지 확인
    # print(len(train))
    # print(len(test))

    # 기존 데이터 유지되는지 확인
    # print(len(train + test) == len(df))
    # print(df)
    test_lst = test.values.tolist()
    train_lst = train.values.tolist()

    k = 5
    tp_cnt=0
    fp_cnt=0
    for i in test_lst:
        result_knn = knn_classify(k, train_lst, i)
        if result_knn == i[5]:
            tp_cnt += 1
        else:
            fp_cnt += 1

    precision = tp_cnt/(tp_cnt+fp_cnt)
    recall = (tp_cnt / len(train))
    f1_score = 2*((precision*recall/precision+recall))

    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print ("F1-score : " + str(f1_score))


if __name__ == '__main__':
    main()
