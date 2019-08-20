from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# アイリスのサンプルデータを読み込み
iris_dataset = load_iris()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'],
    test_size=0.3, random_state=0)

# k-最近傍法でのモデル作成
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print('k-最近傍法での予測結果')
print(score)

# ロジスティック回帰でのモデル作成
lr = LogisticRegression(C=10)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print('ロジスティック回帰での予測結果')
print(score)

# サポートベクターマシンでのモデル作成
svm = SVC()
svm.fit(X_train, y_train)
score = svm.score(X_test, y_test)
print('SVMでの予測結果')
print(score)

# 決定木でのモデル作成
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
score = tree.score(X_test, y_test)
print('決定木での予測結果')
print(score)
