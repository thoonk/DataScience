from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier

w = load_wine()
# print(wine.feature_names)
# print(wine.data[:2])
# print(wine.target_names)
# print(wine.target[:2])
dct = DecisionTreeClassifier(criterion="entropy", random_state=0)
dct.fit(w.data, w.target)
print("입력 데이터의 와인 종류 = " + str(dct.predict([[0.4, 10.7, 20.3, 10.6, 12, 2.8, 3, 0.1, 2.5, 5.1, 1.0, 3.2, 123]])))
print("입력 데이터의 예측 확률 = " + str(dct.predict_proba([[0.4, 10.7, 20.3, 10.6, 12, 2.8, 3, 0.1, 2.5, 5.1, 1.0, 3.2, 123]])))
print("의사결정 나무의 리프의 수 = " + str(dct.get_n_leaves()))
print("의사결정 나무의 높이(깊이) = " + str(dct.get_depth()))
