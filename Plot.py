# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

#图表显示
class Plot():

    def showAll(self,titanicData):
        fig = plt.figure()
        fig.set(alpha=0.2)

        plt.subplot2grid((2,3),(0,0))#创建一个2*3的矩阵图形,在第1行的第1个
        titanicData.Survived.value_counts().plot(kind = 'bar')#bar-垂直柱状图,line-线形图,kde-曲线图,barh-水平柱状图
        plt.title(u"获救情况(1为获救)")#标题
        plt.ylabel(u"人数")#Y轴文字


        plt.subplot2grid((2,3),(0,1))#创建一个2*3的矩阵图形,在第1行的第2个
        titanicData.Pclass.value_counts().plot(kind="bar")
        plt.ylabel(u"人数")
        plt.title(u"乘客等级分布")

        plt.subplot2grid((2,3),(0,2))
        plt.scatter(titanicData.Survived, titanicData.Age)
        plt.ylabel(u"年龄")                         # 设定纵坐标名称
        plt.grid(b=True, which='major', axis='y')
        plt.title(u"按年龄看获救分布 (1为获救)")


        plt.subplot2grid((2,3),(1,0), colspan=2) #横跨两个列
        titanicData.Age[titanicData.Pclass == 1].plot(kind='kde')
        titanicData.Age[titanicData.Pclass == 2].plot(kind='kde')
        titanicData.Age[titanicData.Pclass == 3].plot(kind='kde')
        plt.xlabel(u"年龄")# plots an axis lable
        plt.ylabel(u"密度")
        plt.title(u"各等级的乘客年龄分布")
        plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


        plt.subplot2grid((2,3),(1,2))
        titanicData.Embarked.value_counts().plot(kind='bar')
        plt.title(u"各登船口岸上船人数")
        plt.ylabel(u"人数")
        plt.show()

    def showByPclass(self,titanicData):
        # 看看各乘客等级的获救情况
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数

        Survived_0 = titanicData.Pclass[titanicData.Survived == 0].value_counts()
        Survived_1 = titanicData.Pclass[titanicData.Survived == 1].value_counts()
        df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
        df.plot(kind='bar', stacked=True)
        plt.title(u"各乘客等级的获救情况")
        plt.xlabel(u"乘客等级")
        plt.ylabel(u"人数")
        plt.show()

    def showBySex(self,titanicData):
        # 看看各性别获救情况
        fig = plt.figure()
        fig.set(alpha=0.2)

        Survived_0 = titanicData.Survived[titanicData.Sex == 0].value_counts()
        Survived_1 = titanicData.Survived[titanicData.Sex == 1].value_counts()
        df = pd.DataFrame({u'男性：': Survived_1, u'女性': Survived_0})
        df.plot(kind='bar', stacked=True)
        plt.title(u"各性别获救情况")
        plt.xlabel(u"乘客性别")
        plt.ylabel(u"人数")
        plt.show()

    def showByPclassAndSex(self,titanicData):
        # 各种舱级别情况下各性别的获救情况
        fig = plt.figure()
        fig.set_alpha(0.5)
        plt.title(u'各种舱级别情况下各性别的获救情况')

        ax1 = fig.add_subplot(141)
        titanicData.Survived[titanicData.Sex == 0][titanicData.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                label="female highclass",
                                                                                                color='#FA2479')
        ax1.set_xticklabels([u'获救', u'未获救'], rotation=0)
        plt.legend([u'女性/高级舱'], loc='best')

        ax2 = fig.add_subplot(142, sharey=ax1)
        titanicData.Survived[titanicData.Sex == 0][titanicData.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                label='female, low class',
                                                                                                color='pink')
        ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
        plt.legend([u"女性/低级舱"], loc='best')

        ax3 = fig.add_subplot(143, sharey=ax1)
        titanicData.Survived[titanicData.Sex == 1][titanicData.Pclass != 3].value_counts().plot(kind='bar',
                                                                                                label='male, high class',
                                                                                                color='lightblue')
        ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
        plt.legend([u"男性/高级舱"], loc='best')

        ax4 = fig.add_subplot(144, sharey=ax1)
        titanicData.Survived[titanicData.Sex == 1][titanicData.Pclass == 3].value_counts().plot(kind='bar',
                                                                                                label='male low class',
                                                                                                color='steelblue')
        ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
        plt.legend([u"男性/低级舱"], loc='best')

        plt.show()