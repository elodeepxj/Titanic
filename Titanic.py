# -*- coding:utf-8 -*-
import pandas as pd;
from sklearn import linear_model

from Plot import Plot

from sklearn.ensemble import RandomForestRegressor as df, RandomForestRegressor

titanicData = pd.read_csv("F:/kaggle/train.csv");
print "表信息：",titanicData.shape;
print "表所有字段:",titanicData.columns;
print "字段属性为int的："
print titanicData.describe()
# PassengerId-乘客ID
# Survived-是否获救
# Pclass-乘客等级
# Age-年龄
# SibSP-兄弟姐妹个数
# Parch-父母小孩个数
# Fare-票价
# Name-姓名
# Sex-性别
# Ticket-船票信息
# Cabin-船舱
# Embarked-登船港口


# 冗余数据处理
def redundancyHandler(titanicData):
    # 删除乘客ID
    titanicData = titanicData.drop(["PassengerId"],1)
    # 删除船票信息和船舱信息
    titanicData = titanicData.drop(["Ticket"],1)

    # Sex字段二值化,男性为1，女性为0
    titanicData.loc[titanicData['Sex'] == 'male','Sex'] = 1
    titanicData.loc[titanicData['Sex'] == 'female','Sex'] = 0

    #以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,’no’]，而将其平展开为’Cabin_yes’,’Cabin_no’两个属性
    dummies_cabin = pd.get_dummies(titanicData['Cabin'],prefix='Cabin')
    titanicData =pd.concat([titanicData,dummies_cabin],axis=1)
    titanicData.drop(['Cabin'],1)

    # Age字段缺失值处理
    # 用中位数填充缺失值
    # titanicData['Age'] = titanicData['Age'].fillna(titanicData['Age'].median())

    # 登船港口Embarked字段数字化，S=0，C=1，Q=2
    titanicData['Embarked'] = titanicData['Embarked'].fillna('S')
    titanicData.loc[titanicData['Embarked'] == 'S','Embarked'] = 0
    titanicData.loc[titanicData['Embarked'] == 'C','Embarked'] = 1
    titanicData.loc[titanicData['Embarked'] == 'Q','Embarked'] = 2

    # 用二元制代替 Embarked_C,Embarked_Q,Embarked_S
    # dummies_titanic_data = pd.get_dummies(titanicData.Embarked)
    # dummies_titanic_data = dummies_titanic_data.rename(columns=lambda x:'Embarked_'+str(x))
    # titanicData = pd.concat([titanicData,dummies_titanic_data],axis=1)

    # 数字化和二元制，二选一
    print titanicData.describe()
    return titanicData;

#数据分析
#图表对象，可以用于显示数据图
plot = Plot();


#丢失数据处理
def setMissingAges():
    #添加丢失数据，年龄
    age_df = titanicData[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)


    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    titanicData.loc[(titanicData.Age.isnull()), 'Age'] = predictedAges

    return titanicData, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

#添加丢失数据，年龄
data_train, rfr = setMissingAges()
titanicData = set_Cabin_type(data_train)

# 冗余信息处理
titanicData = redundancyHandler(titanicData)

def buildingModel(titanicData):
    # 用正则取出我们要的属性值
    train_df = titanicData.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_.*|Embarked|Sex|Pclass')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    return clf;

























