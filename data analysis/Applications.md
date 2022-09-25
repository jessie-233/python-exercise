# Application examples

- 处理缺失值
```python
# 检查
pitchfork[pitchfork.isna().any(axis=1)] # 有NaN的行（全表呈现）
# 删除
pitchfork = pitchfork.dropna(subset=["col1", "col2"]).reset_index(drop=True)
```

- 检查numeric列的分布，异常值
```python
# 画分布图
numeric_cols = ['score', 'releaseyear', 'danceability', 'energy', 'key',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, figsize=(10, 7))
for ax, feature in zip(axes.flat, numeric_cols):
    ax.hist(pitchfork[feature],bins=10)
    ax.set_title(feature)
    ax.set_yscale('log') # 重要，不然看不出来异常值！
fig.tight_layout()
# 从图看出端倪后，进一步检查
pitchfork['key'].value_counts(dropna=False).sort_index()
# 删除异常值的行（-1为异常值）
pitchfork = pitchfork.loc[~((pitchfork[tmp] == -1).any(axis=1))]
```

- 检查categorical列的分布，异常值
```python
# 检查
pitchfork['genre'].value_counts(dropna=False)
# 修复
pitchfork = pitchfork.loc[~(pitchfork['genre'] == 'none')]
```

- Regression model构建
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(
     pitchfork[features], pitchfork['score'], test_size=0.3, random_state=123)

# z-score, scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg = LinearRegression() 
# reg = DecisionTreeRegressor(max_depth=5, random_state=17) 
reg.fit(X_train_scaled, y_train_scaled)
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))
```

- Classification model构建
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
     pitchfork[features], pitchfork['score'], test_size=0.3, random_state=123)

# z-score, scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=10, random_state=17)
# clf = KNeighborsClassifier(n_neighbors=10)
# clf = GradientBoostingClassifier()
# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17,max_depth,max_features..)
clf.fit(X_train_scaled, y_train_scaled)

## Evaluation:
# prediction quality:
from sklearn.metrics import accuracy_score, r2_score
y_pred = clf.predict(X_test_scaled)
print('accuracy: ',accuracy_score(y_test, y_pred),'R2: ',r2_score(y_test, y_pred)
```

- cross-validation to select best parameters
```python
# regular method: loop
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

cv_scores, test_scores = [], []
n_neighb = [1, 2, 3, 5] + list(range(50, 550, 50))
for k in n_neighb:
    knn_pipe = Pipeline(
        [("scaler", StandardScaler()), 
        ("knn", KNeighborsClassifier(n_neighbors=k))]
    )
    cv_scores.append(np.mean(cross_val_score(knn_pipe, X_train, y_train, cv=5)))
    # cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5, scoring=默认accuracy https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
    knn_pipe.fit(X_train, y_train)
    test_scores.append(accuracy_score(y_test, knn_pipe.predict(X_test)))

plt.plot(n_neighb, cv_scores, label="CV")
plt.plot(n_neighb, test_scores, label="test_scores")
plt.title("Easy task. kNN fails")
plt.legend()

# GridSearchCV method
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier(random_state=17)
tree_params = {"max_depth": range(1, 11), "max_features": range(4, 19)}
tree_grid = GridSearchCV(tree, tree_params, cv=10, scoring=None, n_jobs=-1, verbose=True)
# scoring默认为None，用的是estimator的误差估计函数，其他可选：
# https://blog.csdn.net/qq_41076797/article/details/102755893
tree_grid.fit(X_train, y_train)

# attributes: .cv_results_,.best_params_,.best_score_,.scorer_
# method: .predict()用最好的para预测
def bootstrap_ci(x,ci=90):
    x = np.array(x)
    mean_vals = [np.mean(np.random.choice(x,len(x))) for _ in range(1000)]
    return np.percentile(mean_vals, 100-ci), np.percentile(mean_vals, ci)

def get_bootstrap_ci():
    # .cv_results_展示所有结果详情
    tmp = pd.DataFrame(tree_grid.cv_results_)
    # 每个fold的结果列名
    target_col = list(tmp.columns[tmp.columns.str.contains('split[0-9]*_test_score', regex=True)])
    # matrix，每个元素是一个K-fold结果list
    all_scores = tmp[['param_max_depth','param_max_features']].join(pd.Series(tmp[target_col].values.tolist(),name='score'))
    all_scores = all_scores.groupby(['param_max_depth','param_max_features'])['score'].sum().unstack(level=-1)
    # apply bootstrap_ci func
    ci_scores = all_scores.applymap(bootstrap_ci)
    # get lower bound and upper bound
    lower_bound = ci_scores.applymap(lambda x: x[0])
    upper_bound = ci_scores.applymap(lambda x: x[1])
    # plot
    fig, axs = plt.subplots(ncols=2, figsize=(25,7))
    sns.heatmap(lower_bound,annot=True,fmt='.3f',ax=axs[0])
    sns.heatmap(upper_bound,annot=True,fmt='.3f',ax=axs[1])
    # for only mean score of every combination:
    mean_scores = tmp[['mean_test_score','param_max_depth','param_max_features']].groupby(['param_max_depth','param_max_features'])['mean_test_score'].sum().unstack(level=-1)
    plt.figure(figsize=(13,7))
    sns.heatmap(mean_scores,annot=True,fmt='.3f')
```
- 整个流程
```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'clf__alpha': [1e-4]
    'clf__max_iter' : [5,10,20]
}

pipe = Pipeline([
    # ("scaler", StandardScaler()),
    ('vect', CountVectorizer()), # Convert a collection of text documents to BOW (counts)
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(penalty='l2', loss='log', max_iter=5, tol=None, random_state=42))
])

text_clf = GridSearchCV(pipe, param_grid=param_grid,scoring=None, cv=5)
text_clf = text_clf.fit(X_train, y_train)

# GridSearchCV attributes: .cv_results_,.best_params_,.best_score_,.scorer_,best_estimator_
# GridSearchCV method: .predict()用最好的para预测
print(text_clf.best_score_)
for param_name in sorted(param_grid.keys()):
    print("%s: %r" % (param_name, text_clf.best_params_[param_name]))

# test set performance
predicted = text_clf.predict(X_test)
print("Accuracy on Test Data: ", accuracy_score(y_test,predicted))

# 解释性
top5_coeff_indices = np.argsort(-text_clf.best_estimator_.named_steps['clf'].coef_)[:,:5]
print(top5_definitions)
np.array(gs_clf.best_estimator_.named_steps['vect'].get_feature_names())[top5_coeff_indices]
