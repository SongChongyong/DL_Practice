# -*- coding: utf-8 -*- 
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree                
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'./AllElectronics.csv', 'rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()    #.next读取csv第一行的数据
print("打印CSV文件的第一行如下")
print(headers)
print("\n")

#sklearn要求读取数据必须是数值型的值，不能读取categorical data(分类数据)(如youth,high,no等)
featureList = []    #创建一个list包含实例age,income,student,sredit_rating的数据
labelList = []          #创建一个list包含Class_buys_computer的数据

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)      #把每一行转化成一个list,如[{'credit_rating': 'fair', 'age': 'youth', 'student': 'no', 'income': 'high'}为第一行的lish
print('\n')

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()  #转换featureList的字典类型数据转换成1和0

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()   
dummyY = lb.fit_transform(labelList)        #把class_by-computers也转换成1和0
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')      #指明决策树标准用的entropy(信息熵)
clf = clf.fit(dummyX, dummyY)           
print("clf: " + str(clf))
print('\n')

# Visualize model  用Graphviz可视化决策树
with open("allElectronicInformationGainOri.dot", 'w') as f:   #存储为allElectronicInformationGainOri.dot文件
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]    #第一行
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1          #第一行中第0列改成1
newRowX[2] = 0          #第一行中第0列改成0
print("newRowX: " + str(newRowX))
print("\n")

predictedY = clf.predict(newRowX)    #预测newRowX情况下Y是1还是0，即买不买电脑
print("predictedY: " + str(predictedY))


