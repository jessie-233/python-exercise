# Data Analysis Pipeline (pandas and numpy)

## 一、提前设置

## 二、读数据进pandas

- 从csv读
[``pandas.read_csv()``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html?highlight=pd%20read_csv)

```python
import pandas as pd
import os

PATH_IN = './data/'
fname = os.path.join(PATH_IN, 'interactive_data.csv')
df = pd.read_csv(fname,sep='\t',header=0,index_col=0,nrows=100,decimal=',')
```

- 从txt读

## 三、缺失值
- 检查数据

```python
df.head() # 打印前5条数据
df.info()
df.describe()
```

```python
# 所有列名，对应非空行数（检查np.nan），Dtype
>>>df.info() 
<class 'pandas.core.frame.DataFrame'>
Int64Index: 540 entries, 1 to 540
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   Intent      540 non-null    object 
 1   Gender      540 non-null    object 
 2   Age         540 non-null    object 
 3   Race        540 non-null    object 
 4   Deaths      540 non-null    int64  
 5   Population  540 non-null    int64  
 6   Rate        468 non-null    float64
dtypes: float64(1), int64(2), object(4)
memory usage: 33.8+ KB
```

```python
# 仅对numeric的列统计 (检查np.inf, -np.inf, 离谱值)
>>>df.describe() 
	Deaths	Population	Rate
count   540.00	    5.40e+02	468.00
mean    995.52	    2.34e+07	inf
std     3322.54	    4.50e+07	NaN
min     0.00	    0.00e+00	0.00
25%     1.00	    1.43e+06	0.10
50%     22.00	    6.86e+06	0.45
75%     191.75	    2.49e+07	3.40
max     33599.00    3.16e+08	inf
```

[``DataFrame.isna()``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html#pandas.DataFrame.isna) 返回与DataFrame同形状的true/false

```python
df['col'].isna() # 返回boolean Series. Characters such as empty strings '' or numpy.inf are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True)
df['col'].isna().sum() # NaN的数量
df.isna().any(axis=1) # 返回boolean Series，所有列，只要有NaN就返回True
df[df.isna().any(axis=1)] # 有NaN的行（全表呈现）
df[~(df.isna().any(axis=1))] # 没有NaN的行（全表呈现）
np.array(df.index)[df.isna().any(axis=1)] # 打印NaN行的index
```
除了检查NaN，还要检查异常值

[``DataFrame.value_counts(subset=None, normalize=False, sort=True, ascending=False, dropna=True)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.value_counts.html?highlight=value_counts#pandas.DataFrame.value_counts)统计此列出现过的值及次数

[``Series.unique()``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html?highlight=unique)返回一个Series里所有不同值


```python
# 检查object列异常值（统计此列出现过的值及次数）
>>>df['col'].value_counts(dropna=False) 
None selected    108
Suicide          108
Homicide         108
Accident         108
Unknown          108
Name: Intent, dtype: int64
# 也可检查numeric异常值
df['col'].value_counts(dropna=False).sort_index()
sort(df['col'].value_counts(bins=4,dropna=False).to_dict().items()) # bins将numeric数值平均分组
# 检查numeric异常值另一种方法（画分布图）
df['col'].hist()
```

```python
# 返回和打印异常值
# df['col'] == '异常'-->返回一个Boolean pandas Series
# ~(df['col'] == '异常')-->返回与上相反

# 共有多少异常行
sum(df['col'] == '异常') 
sum(df['col'] == np.inf)

# 打印
df[df['col']=='异常'] # 打印异常的行（全表呈现）
df.loc[~(df['col'] == '异常')] # 不异常的行（全表呈现）
df.loc[~(df=='异常').any(axis=1)] # 同上，此‘异常’在多列出现
np.array(df.index)[df['col'] == np.inf] # 打印异常行的index
```

- 处理
1. 处理NaN：

删除：
[``DataFrame.dropna(axis=0,how='any',thresh=None,subset=None,inplace=False)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)

填充: 
[``DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)

```python
'''
axis：drop行还是drop列
how：只要有一个NaN就drop（any），所有全是NaN才drop（all）
subset：针对哪一列的NaN-->subset=['col1','col2']
inplace：True-->do operation inplace and return None
'''
# drop并重置索引
df = df.dropna(subset=["col1", "col2"]).reset_index(drop=True)
# fillna

```

2. 处理inf，-inf等异常值：

方法一：先替换成np.nan，再dropna

[``DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace)

[``DataFrame.where(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.where.html?highlight=where#pandas.DataFrame.where)Replace values with 'other' where the condition is False.

[``DataFrame.mask(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mask.html?highlight=mask#pandas.DataFrame.mask)Replace values with 'other' where the condition is True.

```python
df = df.replace(['异常值1', '异常值2'], np.nan).dropna(subset=["col1", "col2"]).reset_index(drop=True)
# or
df = df.where(~(df == '异常值1' | df == '异常值2'), other=np.nan).dropna(subset=["col1", "col2"]).reset_index(drop=True)
# or
df.mask(df == '异常值1' | df == '异常值2', other=np.nan).dropna(subset=["col1", "col2"]).reset_index(drop=True)
```
方法二：检索-->过滤

```python
df_filtered = df.loc[~(df=='异常值').any(axis=1)]
```
## 四、重复值
```python
# check if the names are unique
len(df['Name'].unique()) == len(df)
# Calculates duplicated rows
# df[df.duplicated(subset=["id"])].index
num_duplicates = len(df[df.duplicated(subset=["id"])])
print("There were {} duplicated rows".format(num_duplicates))
# Removes duplicated rows
df = df.drop_duplicates(subset=["id"], keep="first")
```

## 五、数据分析基本操作

### pandas增删改查
1. 增

- 增加列
[``DataFrame.insert(loc, column, value, allow_duplicates=False)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace)
```python
# OR（增加列）
df.loc[:,'new_col'] = ...
```
- 两表合并
  
[``DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html?highlight=join#pandas.DataFrame.join)

最好用merge:
[``DataFrame.merge(right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html?highlight=merge#pandas.DataFrame.merge)
```python
# 默认joins index-on-index
df_left.join(df_right,rsuffix='_right')
# joins on col
df_left.join(df_right.set_index('col'),on='col')
happiness.join(countries.rename({'country_name':'country'},axis=1).set_index('country'),on='country')
# merge
country_features = happiness.merge(countries, on="country")
```

1. 删
- 删除行/列
[``DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace)

```python
X = df.drop(["Churn", "State"], axis=1)
```

3. 改
```python
# 修改某一列类型.astype()/to_datetime()
df.Gender = df.Gender.astype('category')
df["User_Score"] = df["User_Score"].astype("float64")
df["Year_of_Release"] = df["Year_of_Release"].astype("int64")
df['date'] = pd.to_datetime(df['date'], format='%B %d %Y') # format见(https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
Series.dt.year() # 返回year
# 查看Dtype
df.dtypes
# 修改列名
df = df.rename({'Historical Significance': 'Role'},axis=1)
# 设置某一列为索引
df.set_index('Name', inplace=True)
# 还原索引，重新变为默认的整型索引 
df.reset_index(drop=False, inplace=False) # drop为False则索引列会被还原为普通列，否则会丢失
# 某一列转换为字典：{索引：值}
df['Role'].to_dict()
# 将某列中的字符串统一修改Series.str将每个str元素迭代
>>>a = pd.Series(['apple','melon','pear'])
>>>a.str.startswith('a') # 返回boolean series True False False
>>>a.str.replace('a','A') # 返回boolean series Apple melon peAr

# 按条件替换值.where(cond, other=nan) 与.mask(cond, other=nan)相反
# 将yes/no等分类变量替换为0/1/2...pd.factorize()
```
4. 查
- 索引
```python
## 列索引
df['col']
df[['col1','col2']]
## 行索引
# boolean
df[(df["col1"] == condition1) & (df["col2"] == condition2))]
df[df["col1"].isin(['a','b'])] # .isin()函数返回Booleans Series
# 指定index
df[:1] # 第一行（左闭右开）
df[2:3] # 第三行（左闭右开）
df[-1:] # 最后一行
## 行、列共同索引
# loc:indexing by name
df.loc[0:5, "col1":"col2"] # index从0到5（包含），列从"col1"到"col2"（包含）
df.loc[0:5, ["col1","col2"]] 
# iloc:indexing by number
df.iloc[0:5, 0:3] # 左闭右开
```
- 迭代每行数据：df.iterrows()
```python
for idx, row in country_features.sort_values('literacy',ascending=False).iterrows():
    print(row.world_region, row.country, row.happiness_score)
```
- Series.index返回index类型，需要用list(Series.index[:top_n])转换
- 条件判断
  
[``Series.str.startswith(pat, na=None)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.startswith.html?highlight=str%20startswith#pandas.Series.str.startswith)判断每一个字符串元素都以..开头
``Series.str.endswith(pat, na=None)``判断每一个字符串元素都以..结尾
``Series.str.contains(pat, case=True, flags=0, na=None, regex=True)``判断每一个字符串元素是否包含...

```python
data['marital-status'].str.startswith('Married') # 返回Booleans Series
df["col1"].isin(['a','b']) #
```

- 某一列求和/求平均
```python
df.sum() # 会对所有numeric列求和
df['col'].sum() # .sum()是对pd.Series的操作，对list用sum(list)
df['col'].mean()
df['col'].max()
df['col'].min()
result = df[
            (df['col1'] == 'condition1') & 
            (d['col2'] == 'condition2')
            ]['col'].sum() # 先对某些列做筛选

```
- Series.values返回ndarray
- 按某些列排序
[``DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html?highlight=sort_values#pandas.DataFrame.sort_values)

```python
'''
by：按哪些列排序-->['col1','col2']
ascending：升序
na_position：若有na，排在最前还是后
ignore_index：排序后重置index，默认不重置
'''
df = df.sort_values(by=['col1', 'col2'],ascending=False,na_position='first',ignore_index=True)
```
- 对行、列施加函数
  
[``DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html?highlight=apply#pandas.DataFrame.apply)

[``Series.apply(func, convert_dtype=True, args=(), **kwargs)``](https://pandas.pydata.org/docs/reference/api/pandas.Series.apply.html?highlight=apply#pandas.Series.apply)

[``Series.map(arg, na_action=None)``](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html?highlight=map#pandas.Series.map)

[``DataFrame.agg(func=None, axis=0, *args, **kwargs)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html?highlight=agg)

```python
'''
func：要施加的函数，可以是lambda x: func(x)
axis：函数施加在列（0），行（1）。默认列
'''
>>>df = pd.DataFrame([[1, 2],[3,4],[5,6]], columns=['A', 'B'])
        A	B
0	1	2
1	3	4
2	5	6
>>>df.apply(np.sum, axis=0) # 每一列求和
A     9
B    12
dtype: int64
>>>df.apply(np.sum, axis=1) # 每一行求和
0     3
1     7
2    11
dtype: int64
>>>df.apply(lambda x: x[0]>4, axis=0) #列
A    False
B    False
dtype: bool
>>>df.apply(lambda x: x[0]>4, axis=1) #行
0    False
1    False
2     True
dtype: bool
# 实用小技巧：用.groupby().apply()生成一些想要的列（dataframe）
>>>stats_by_year = movies.groupby(['year']).apply(lambda x: pd.Series({
        'average_worldwide_gross': x['worldwide_gross'].mean(),
        'lower_err_worldwide_gross': bootstrap_CI(x['worldwide_gross'], 1000)[0],
        'upper_err_worldwide_gross': bootstrap_CI(x['worldwide_gross'], 1000)[1]
    }))  # x代表每行（每year_index）

# Series.apply对每个元素进行func
df['col'].apply(np.mean)
df['col'].apply(lambda x: x[0] == "W") # x代指此列（series）中每行的值，返回Boolean series：col列中每个值是否以W开头

# Series.map可以对每个元素进行不同处理
>>>s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
0       cat
1       dog
2       NaN
3    rabbit
dtype: object
>>>s.map({'cat': 'kitten', 'dog': 'puppy'}) # 字典中不存在的默认全变NaN（也可用replace实现）
0    kitten
1     puppy
2       NaN
3       NaN
dtype: object
>>>s.map('I am a {}'.format)
0       I am a cat
1       I am a dog
2       I am a nan
3    I am a rabbit
dtype: object
>>>s.map('I am a {}'.format, na_action='ignore') # 忽略na项
0       I am a cat
1       I am a dog
2              NaN
3    I am a rabbit
dtype: object
# .agg()
>>>df.agg({'A':['sum','min'], 'B':['min','max']})
	A	B
sum	9.0	NaN
min	1.0	2.0
max	NaN	6.0
```
- 找到每列最大（小）值的index，返回series
```python
df.apply(lambda x: x.idxmax(), axis=0) # idxmin()
```
- 找到每行最大（小）值的column name，返回series
```python
df.apply(lambda x: x.idxmax(), axis=1) # idxmin()
```
- 分组

[``DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html?highlight=groupby#pandas.DataFrame.groupby)
通常用法``df.groupby(by=grouping_columns)[columns_to_show].function().to_frame()``（若不指定``[columns_to_show]``，则显示所有列结果）

[``pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html?highlight=crosstab#pandas.crosstab)数据透视表


```python
'''
by：按哪列分组-->['col1']
返回一个DataFrameGroupBy对象
'''
>>>l = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
>>>df = pd.DataFrame(l, columns=["a", "b", "c"])
        a	b	c
0	1	2.0	3
1	1	NaN	4
2	2	1.0	3
3	1	2.0	2
>>>df.groupby(by=["b"])['a'].sum().to_frame()
    a   
b
1.0 2   
2.0 2 
>>>df.groupby(by=["b"],dropna=False).agg([np.mean, np.std, np.min, np.max])
	a	                        c
        mean	std	amin	amax	mean	std	amin	amax
b								
1.0	2.0	NaN	2	2	3.0	NaN	3	3
2.0	1.0	0.0	1	1	2.5	0.707107	2	3
NaN	1.0	NaN	1	1	4.0	NaN	4	4
# 解析DataFrameGroupBy对象
>>>from IPython.display import display
>>>for b, sub_df in df.groupby(by=["b"],dropna=False):
    print(b)
    display(sub_df)

1.0
	a	b	c
2	2	1.0	3

2.0
    a	b	c
0	1	2.0	3
3	1	2.0	2

nan
	a	b	c
1	1	NaN	4
```
```python   
>>>pd.crosstab(df["b"], df["a"]) # 默认统计个数
a	1	2
b		
1.0	0	1
2.0	2	0
>>>pd.crosstab(data['native-country'], data['salary'], 
           values=data['hours-per-week'], aggfunc=np.mean) # value要统计的值，aggfunc要对这些值如何操作
```

### numpy增删改查？？

## 六、数据可视化
plt的两个用法：指定figure、axes / 不指定自动生成（）只能一个窗口一幅图
Figure理解成一个窗口，Axes才是真正的图，一个Figure可以有很多Axes
```python
# 指定
x = np.arange(1,10)
y = x
fig = plt.figure() #生成一个figure
axes = plt.subplot() # 生成一个轴
axes.scatter(x,y,c='r',marker='o')
plt.show()
# 指定
# 1行三列
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 7)) 
# 第一列
sns.boxplot(x="Churn", y="calls", data=df, ax=axes[0])
axes[0].set_xlabel("")
axes[0].set_ylabel("calls")
# 第二列
axes[1].plot(x,x**2)
# 第三列
axes[2].scatter(x,x**2)
fig.tight_layout()
# 不指定
plt.hist(movies['worldwide_gross'].values,bins=100,log=True)
```

1. 单变量
- **numeric**

*histogram矩形分布图*
[``DataFrame.hist(bins=100,xlabelsize=10,xrot=45,figsize=(10,5))``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.hist.html?highlight=hist#pandas.DataFrame.hist)
``plt.hist()`` 有range、log参数
[``sns.distplot(df['col'])``](https:/?)带kernel density曲线

```python
# 两个打印到一张图
movies.hist(column=['worldwide_gross','length'],bins=100,xlabelsize=10,xrot=45,figsize=(10,5))
# OR
# sns.histplot(zeros, kde=True, alpha=0.5, label='First album', ax=axs[1])
# sns.histplot(ones,kde=True, alpha=0.5, label='Second album', color='C8', ax=axs[1])
ax = sns.distplot(treated['re78'], hist=True, label='treated')
ax = sns.distplot(control['re78'], hist=True, label='control')
ax.set(title='Income distribution comparison in 1978',xlabel='Income 1978', ylabel='Income density')
plt.legend()
plt.show()
# 一张图一个
movies['worldwide_gross'].hist()
plt.xlabel('Worldwide gross revenue')
plt.ylabel('Number of movies')
plt.title('Gross revenue, histogram')
# 直接用matplotlib
plt.hist(movies['worldwide_gross'].values,bins=100,log=True)
# log-log绘图
plt.loglog()
```

*boxplot箱型图 / violin图*
[``plt.boxplot()``](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html?highlight=boxplot#matplotlib.pyplot.boxplot)
``sns.boxplot(df['col'])``
```python
plt.boxplot(movies['worldwide_gross'])
plt.xticks([]) # disable x tick
----------
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6, 4))
sns.boxplot(data=df["Total intl calls"], ax=axes[0])
sns.violinplot(data=df["Total intl calls"], ax=axes[1])
```

- **categorical**

``sns.countplot()``
```python
sns.countplot(x="Churn", data=df) # "Churn"是分类变量
```

2. 多个变量
- numeric-numeric

``plt.plot()``
``plt.scatter()``不适用于很多数据点-->乱成一团难以解释-->用sns下面
``sns.jointplot()``可以在边上显示单变量histogram
``sns.pairplot()``多个变量的散点图矩阵（对角线是单变量histogram）
``sns.regplot(x='poli_atten_val', y='political_support_rate', data=df)``与``sns.jointplot(kind="reg")``同

```python
# iterate the different groups to create a different series
for country, missions in missions_by_date.groupby("ContryFlyingMission"): 
    plt.plot(missions["MissionDate"], missions["MissionsCount"], label=country)
plt.legend(loc='best')
----------
plt.scatter(movies['worldwide_gross'], movies['imdb_rating'],s = 2) # marker size s=2
plt.xlabel('Worldwide gross revenue')
plt.ylabel('IMDB rating')
# sns
sns.jointplot(movies['worldwide_gross'], movies['imdb_rating'], kind="hex")
sns.jointplot(data = movies, x = 'worldwide_gross', y = 'imdb_rating', kind="kde")
sns.jointplot(data = movies, x = 'worldwide_gross', y = 'imdb_rating', kind="reg")
sns.pairplot(df[['col1','col2','col3']])
```
``sns.heatmap()``
```python
corr_matrix = df[['col1','col2','col3']].corr()
sns.heatmap(corr_matrix,annot=True,fmt='.1f',square=True)
```

- numeric-categorical

``pd.plot(kind="bar")``
``sns.barplot()``
``plt.barh(x, y, alpha=0.6)``水平的bar
``sns.boxplot()``
``sns.violinplot()``
``sns.lmplot()`` 横纵坐标两个numeric，hue控制类别

```python
pd.plot(kind="bar",x="ContryFlyingMission", y="MissionsCount",figsize=(10,7),log=True,alpha=0.5,color="olive")
sns.barplot(x="Main_Genre", y="worldwide_gross", data=movies.loc[movies['Main_Genre'].isin(['Thriller','Comedy','Fantasy','Sci-Fi','Romance'])])
sns.boxplot(x="Main_Genre", y="worldwide_gross", data=movies.loc[movies['Main_Genre'].isin(['Thriller','Comedy','Fantasy','Sci-Fi','Romance'])])
sns.violinplot(x="Main_Genre", y="worldwide_gross", data=movies.loc[movies['Main_Genre'].isin(['Thriller','Comedy','Fantasy','Sci-Fi','Romance'])])
# sns.lmplot()
sns.lmplot('SelfEmployed','IncomePerCap', data=SetA_per_capita_self_empl, hue = 'State')
plt.xlabel("Percentage of Self Employed people [%]")
plt.ylabel("Income per Capita [$]")
plt.ylim([10000,50000])
plt.xlim([0,22])
```
用``plt.``控制输出
```python
plt.ylim(min,max)
plt.xlabel('Worldwide gross revenue')
plt.ylabel('Number of movies')
plt.title('Gross revenue, histogram')
plt.figure(figsize=(25,10))
plt.xticks(np.linspace(0,1000,15,endpoint=True),rotation=45)  # 设置x轴刻度
plt.xticks([0,1],['control','treat'],fontsize=9,rotation=45) # 设置x轴刻度为文字
plt.tight_layout()
```

- categorical-categorical

``sns.countplot(hue=)``
```python
sns.countplot(x="Customer service calls", hue="Churn", data=df)
```

panel控制
```python
## 循环法
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=False, figsize=(10, 7))
for idx, feature in enumerate(numerical_cols):
    ax = axes[int(idx/4), idx%4] # int(idx / 4) = math.floor(i/4)
    sns.boxplot(x="Churn", y=feature, data=df, ax=ax)
    # 其他图
    # ax.hist(sentiment['Score'].values,bins=50)
    # ax.semilogy(data[char].value_counts().values)
    # sns.countplot(x='age_range', hue='Sentiment', data=filtered_data, \
    #           order=['20-35','35-50','50-65','65-90'], \
    #           hue_order=['NEGATIVE','NEUTRAL','POSITIVE'])
    # 对子图x轴旋转
    # ax.tick_params('x', labelrotation=20)
    # OR
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)
    ax.set_title("子图title")
    ax.set_xlabel("")
    ax.set_ylabel(feature)
# 大图y标签
# fig.text(-0.01, 0.5, 'quotes count', va='center', rotation='vertical',fontsize=15)
# 大图标题
# fig.text(0.3, 1, 'How many quotations there are in each category', va='center',fontsize=15)
plt.suptitle('Sentiment Score Distribution (2008-2020)',x=0.5,y=1)
fig.tight_layout()

# OR
for ax, feature in zip(axse.flat, numerical_cols):
    threshold_score[feature].plot(ax=ax, grid = True)

## 逐个轴
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
axes[0].plot(df.year,df.CO2_emission,linewidth=2,color ='steelblue',marker='o',markersize=5)
axes[0].set_title('CO2 Emissions in the U.S. (2008-2019)')
axes[0].set_xticks(np.arange(2008,2021))
axes[0].set_ylabel('CO2 Emissions (MMT CO2 eq.)')

axes[1]....
fig.tight_layout()
```

## 七、数据分析方法

- data采样 (\Tutorials\03 - Describing data)
```python
#make 10 samples with replacement 有放回的
sample1_counties = df.sample(n = 10, replace = True)

#make 10 samples without replacement 无放回的
sample1_counties = df.sample(n = 10, replace = False)

#sometimes we want to sample in an unbalanced way, so that we upsample datapoints of certain characteristic,
#and downsample the others. this can be achieved with weights parameter
#here we sample by upsampling counties with large population
sample2_counties = df.sample(n = 10, replace = False, weights = df['TotalPop'])
```
- one-hot
```python
# 将某列变为onehot并与原表合并，并改名
pitchfork = pitchfork.join(pd.get_dummies(pitchfork['genre']).rename(columns=lambda x: f'{x}_onehot'))
youtube_ml = pd.get_dummies(youtube_ml, columns=['channel'], prefix='channel_')
```
- Bootstrap (\Tutorials\03 - Describing data)
The 95% (or other) confidence interval of the **average height** of 1000 people using bootstrap resampling: Sampling 1000 height values with replacement and computing the mean. This is **repeated 10000 (nbr_draws) times** to create a **sorted list of the 10000 means**. The CI is defined by the 250th (np.nanpercentile(data, 2.5)) and the 9750th (np.nanpercentile(data, 97.5)) value in sorted order.

```python
# Input: data: your array; nbr_draws:实验多少次(e.g., 1000 is a good number)
# Output: lower error, upper error

def bootstrap_CI(data, nbr_draws):
    means = np.zeros(nbr_draws)
    data = np.array(data)
    # 重复采样nbr_draws次（1000）
    for n in range(nbr_draws):
        # 随机选出len(data)个data中的数（可重复）-->（有放回的,replace=True）
        data_tmp = np.random.choice(data,len(data))
        # 计算忽略NaN值的数组平均值
        means[n] = np.nanmean(data_tmp) 

    return np.nanpercentile(means, 2.5),np.nanpercentile(means, 97.5) # np.nanpercentile()找到一组数的分位数值
# 简化版
def bootstrap_ci(data,nbr_draws):
    data = np.array(data)
    mean_vals = [np.random.choice(x,len(x)).mean() for _ in range(nbr_draws)]
    return np.quantile(mean_vals, 0.05), np.quantile(mean_vals, 0.95)
```

- 检验分布 (\Tutorials\03 - Describing data)

```python
from statsmodels.stats import diagnostic
diagnostic.kstest_normal(df['IncomePerCap'].values, dist = 'norm') # Returns: ksstat(float):Kolmogorov-Smirnov test statistic with estimated mean and variance;
# pvalue(float): p-value < 0.05 --> reject the Null hypothesis that the sample comes from a normal distribution.-->not from normal distribution
# how about exponential?
diagnostic.kstest_normal(df['IncomePerCap'].values, dist = 'exp')
```

- 相关系数分析、假设检验 (\Tutorials\03 - Describing data)
```python
from scipy import stats
stats.ttest_ind(df.loc[df['State'] == 'New York']['IncomePerCap'], df.loc[df['State'] == 'California']['IncomePerCap'])
# P-value < 0.05 --> reject null hypothesis --> 两组值期望不一样
# P-value > 0.05 --> accept null hypothesis --> 两组值期望一样
stats.pearsonr(df['IncomePerCap'],df['Employed']) # return: r(float):Pearson’s correlation coefficient;
# p-value(float):Two-tailed p-value. if <0.05 --> significant
stats.spearmanr(df['IncomePerCap'],df['Employed'])

# 检验两组数据的期望是否相同
# null hypothesis: the two independent samples have identical average (expected) values.
stats.ttest_ind(df.loc[df['State'] == 'New York']['IncomePerCap'], df.loc[df['State'] == 'California']['IncomePerCap'])
# P-value < 0.05 --> reject null hypothesis --> 两组值期望不一样
# P-value > 0.05 --> accept null hypothesis --> 两组值期望一样
```
- 数据标准化、随机打乱、训练集分割

```python
## 标准化
# 调包
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train) # 返回均值，方差ndarray
X_train_std = scaler.transform(X_train) # 返回ndarray而不是dataframe
X_test_std =  scaler.transform(X_test) # 对测试集同样转换
# OR
import scipy.stats as stats
stats.zscore(df,axis=0) # 计算每列
# 手动
means = X_train.mean()
stddevs = X_train.std()
X_train_std = X_train.copy()
X_train_std = (X_train_std - means) / stddevs
X_test_std = X_test.copy()
X_test_std = (X_test_std - means) / stddevs
# df['age'] = (df['age']-df['age'].mean())/df['age'].std()

## 随机打乱
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=0)

## 训练集分割
# 调包
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
# 手动
def split_set(data_to_split, ratio=0.8):
    mask = np.random.rand(len(data_to_split)) < ratio # 返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)
    return [data_to_split[mask].reset_index(drop=True), data_to_split[~mask].reset_index(drop=True)]
[train, test] = split_set(data_features)
```

- 回归分析 (04-Regression analysis; 06-Supervised Learning; 完整:07-Applied ML)
```python
## Linear Regression
import statsmodels.formula.api as smf
'''
    1. `~` : Separates the left-hand side and right-hand side of a formula.
    2. `+` : Creates a union of terms that are included in the model.
    3. `:` : Interaction term. a×b
    4. `*` : `a * b` is short-hand for `a + b + a:b`, and is useful for the common case of wanting to include all interactions between a set of variables.
- Intercepts are added by default. 
- Categorical variables can be included directly by adding a term C(a)
- two standard errors approximate the CIs
'''
mod = smf.ols(formula='time ~ C(high_blood_pressure) * C(DEATH_EVENT,  Treatment(reference=0)) + C(diabetes)',data=df)
res = mod.fit()
print(res.summary()) # p-value小于0.05说明这个predictor是significant的
## Logistic Regression: binary y
# stantardization
df['age'] = (df['age'] - df['age'].mean())/df['age'].std()
mod = smf.logit(formula='DEATH_EVENT ~  age + creatinine_phosphokinase + ejection_fraction + \
                        platelets + serum_creatinine + serum_sodium + \
                        C(diabetes) + C(high_blood_pressure) +\
                        C(sex) + C(anaemia) + C(smoking) + C(high_blood_pressure)', data=df)
res = mod.fit()
print(res.summary())

# visualize the effect of all the predictors
variables = res.params.index
coefficients = res.params.values
p_values = res.pvalues
standard_errors = res.bse.values
l1, l2, l3= zip(*sorted(zip(coefficients[1:], variables[1:], standard_errors[1:]))) # sort只能用于list，sorted可以对所有可迭代对象进行排序
plt.errorbar(l1, np.array(range(len(l1))), xerr= 2*np.array(l3), linewidth = 1,
             linestyle = 'none',marker = 'o',markersize= 3,
             markerfacecolor = 'black',markeredgecolor = 'black', capsize= 5)
plt.vlines(0,0, len(l1), linestyle = '--')
plt.yticks(range(len(l2)),l2)
# log odds of Logistic Regression: an increase of age by 1 standard deviation
# leads on average to an increase by 0.66 of log odds of death

## Regression with sklearn
# Linear
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales
lin_reg = LinearRegression()  # create the model
lin_reg.fit(X, y)  # train it
print(lin_reg.coef_[0], lin_reg.intercept_)
# predict:
lr = LinearRegression()
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, X, y, cv=5)
# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()
print(mean_squared_error(y, predicted))

# train with regularization (reduce model complexity)
ridge = Ridge(alpha=6)
predicted_r = cross_val_predict(ridge, X[:5], y[:5], cv=5)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(y, predicted_r, edgecolors=(0, 0, 0))
ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
ax.set_xlabel('Original')
ax.set_ylabel('Predicted')
plt.show()

# Logistic
logistic = LogisticRegression(solver='lbfgs', penalty='l2', C=0.1)
logistic.fit(X, y)
logistic.predict([test]) # array([0], dtype=int64) 0-->die
logistic.predict_proba([test]) # array([[0.5528599, 0.4471401]]) die
# precision, recall, roc, auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import auc, roc_curve
logistic = LogisticRegression(solver='lbfgs')
precision = cross_val_score(logistic, X, y, cv=10, scoring="precision")
recall = cross_val_score(logistic, X, y, cv=10, scoring="recall")
# Precision: avoid false positives
print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))
# Recall: avoid false negatives
print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))

# Predict the probabilities with a cross validationn
y_pred = cross_val_predict(logistic, X, y, cv=10, method="predict_proba")
# Compute the False Positive Rate and True Positive Rate
fpr, tpr, _ = roc_curve(y, y_pred[:, 1])
# Compute the area under the fpr-tpr curve
auc_score = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Area = {:.5f}".format(auc_score))
```

```python
# crossval找最佳参数，然后用最佳参数训练
accs = []
#the grid of regularization parameter 
grid = [0.01,0.1,1,10,100,1000,10000]
for c in grid:
    #initialize the classifier
    clf = LogisticRegression(random_state=0, solver='lbfgs',C = c)
    #crossvalidate
    scores = cross_val_score(clf, X_train,Y_train, cv=10) # 默认是accuracy（改：scoring="precision"）
    accs.append(np.mean(scores))
plt.plot(accs)
plt.xticks(range(len(grid)), grid)
plt.xlabel('Regularization parameter \n (Low - strong regularization, High - weak regularization)')
plt.ylabel('Crossvalidation accuracy')
plt.ylim([0.986,1]) # 从图中选出最优grid值为10
# train on entire training set
clf = LogisticRegression(random_state=0, solver='lbfgs',C = 10).fit(X_train,Y_train)
#predict on the test set
print('Accuracy:',clf.score(X_test,Y_test))
# OR
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,clf.predict(X_test))
```
- 机器学习

[``KNN``](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier)(06-Supervised Learning)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
clf = KNeighborsClassifier(10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# precision = cross_val_score(clf, X, y, cv=10, scoring="precision")
# recall = cross_val_score(clf, X, y, cv=10, scoring="recall")
mse = mean_squared_error(y_test, y_pred)
prec = precision_score(y_test, y_pred)
```
[``Random Forest``](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier)(06-Supervised Learning)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
# clf_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=17)
clf.fit(X, y)
```
``clustering:``(08-Unsupervised learning)
[``k-means``](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
[``DBSCAN``](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html?highlight=dbscan)

```python
## K-means
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, random_state=0).fit(X)
plt.scatter(X[:,0], X[:,1], c=model.labels_, alpha=0.6)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
# Plot the centroids
for c in model.cluster_centers_:
    plt.scatter(c[0], c[1], marker="o", color="red")
# y_pred = model.predict(X_test)
# Attributes: cluster_centers_, labels_, inertia_, n_iter_, n_features_in_, feature_names_in_

# OR .fit_predict(X)直接返回label
labels = KMeans(n_clusters=3, random_state=0).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, alpha=0.6)
plt.xlabel("feature 1")
plt.ylabel("feature 2")

## DBSCAN
from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.15).fit(X)

## finding optimal K
# silhouette score (选score最大的对应K)
from sklearn.metrics import silhouette_score
silhouettes = []
for k in range(2, 11): # Try multiple k
    # Cluster the data and assign the labels
    labels = KMeans(n_clusters=k, random_state=10).fit_predict(X)
    # Get the Silhouette score
    score = silhouette_score(X, labels)
    silhouettes.append({"k": k, "score": score}  
silhouettes = pd.DataFrame(silhouettes) # Convert to dataframe
# Plot the data
plt.plot(silhouettes.k, silhouettes.score)
plt.xlabel("K")
plt.ylabel("Silhouette score")

# Elbow method: sum of squared errors (选最小的对应K)
def plot_sse(features_X, start=2, end=11):
    sse = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(features_X)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
```
``PCA, TSNE``:多维数据降维，可以与clustering合用（1.画图时用降维后的坐标，color用聚类结果label；2.先将训练数据降维再聚类，避免维数灾难）
```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
X_reduced_pca = PCA(n_components=2).fit(X).transform(X)
X_reduced_tsne = TSNE(n_components=2, random_state=0).fit_transform(X)
```

## 八、有用知识点
- 格式化输出
```python
# 用f
print(f'{suicides/all_deaths*100}% of gun deaths are suicides.')
# 用%
print("Name:%-2s Age:%-2d Height:%-2.2f Percent:%-2.0f%%" % ("Aviad",25,1.83,98.4))
# 用.format
print("Name:{0:<4} Age:{1:<4} Height:{2:<4.2f} Percent:{3:<4.0%}".format("Aviad",25,1.83,0.984))
```
- 改变dataframe某列数据类型
```python
# "object","int64","float64","bool","datetime64","timedelta[ns]","category"
df["boolean"] = df["boolean"].astype("int64")
```
- numpy索引左闭右开
- 屏蔽warning
```python
import warnings
warnings.filterwarnings("ignore")
```
- list.append() & list.extend()
1. append可以添加单个元素，也可以添加可迭代对象，但是extend只能添加可迭代对象
2. 在添加可迭代对象时，append在添加后不改变添加项的类型，extend在添加后，会将添加项进行迭代，迭代的元素挨个添加到被添加的数组中
- 以表格形式打印dataframe
```python  
from IPython.display import display
display(df[~mask1]) # display Pandas dataframe in a Table
```
- dataframe的深拷贝、浅拷贝（default: deep=True）
```python
# 深拷贝：改变df_1, df_2不受影响
df_2 = df_1.copy()
# 浅拷贝：相当于赋值
df_2 = df_1.copy(deep=False)
# 相当于df_2 = df_1
```

- 将多级索引的series其中一级索引作为列，形成pivot table
```python
# .unstack(level=['级名'])
df.groupby(['Platform','Genre'])['Global_Sales'].sum().unstack(level=-1).fillna(0)
# 直接pivot table方法
df.pivot_table(index="Platform", columns="Genre", values="Global_Sales", aggfunc=sum)
```

- 将字符串转化为列表/元组/字典（原本类型）
```python
import ast
a = "[1,2,3]"
ast.literal_eval(a) #[1,2,3]
```
- 字典排序
```python
sorted_dict = sorted(dict.items(), key=itemgetter(1), reverse=True) # 以value降序排列，返回列表里面套元组
```
