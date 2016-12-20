# coding=utf-8
from __future__ import unicode_literals
from __future__ import division
from sklearn import datasets
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.externals import joblib
import jieba.analyse
import codecs
import numpy as np



#对测试数据中用户搜索词条进行分词
def seg_test_data(finename):
  file_test = open(finename, encoding="utf-8")
  test_seg_array=[]
  for line in file_test.readlines():
     temp=""
     for sentence in line.split()[1:]:
       temp += (sentence + " ")
     seg_list = jieba.cut(temp, cut_all=False)
     seg_content = " ".join(seg_list)
     test_seg_array.append(seg_content)
  return test_seg_array



#已弃用
##TODO: 简单粗暴的解决0属性值问题
#对结果0的简单处理,即先统计该属性(age,gnder,education)的预测值的
#所有的标签(0,1,2,3,4,5,6)，然后将0值替换为统计数最多的那个值
def handle_zero_result(predicted):
    counts=[]
    #统计预测结果为1-6的出现次数
    for i in range(1,7):
      num=predicted.count(i)
      counts.append(num)
    print(counts)

    #得到出现次数最多的次数
    max_count=max(counts)
    #如果出现次数最多的次数值和1-6中的某个值出现的次数值相同
    #则该值（1-6中）即为次数最多的
    for i in range(1, 7):
      if(max_count==predicted.count(i)):
          temp=i

    #替换数组中的0值元素
    j = 0
    for item in predicted:
        if(item == 0):
            predicted[j] = temp
        j += 1
    return predicted


# 获取最优参数 ，对不同模型手动改写即可，这个是针对SVM，通过自己需求调用
def opt_parameter(folder_name):
    age_train_data = datasets.load_files(folder_name)
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    svc_clf = SVC()

    data=age_train_data.data
    target=age_train_data.target
    C = range(1, 10)
    kernel = ['rbf', 'linear','poly','sigmoid']
    #degree=range(1,10)
    param_grid = dict(C=C, kernel=kernel)

    #对测试集和训练集分一共分为10份
    rand = RandomizedSearchCV(svc_clf, param_grid,  n_jobs=-1, cv=10, scoring='accuracy')

    fea_train_counts = count_vect.fit_transform(data)
    train_tfidf = tfidf_transformer.fit_transform(fea_train_counts)
    rand.fit(train_tfidf, target)

    print(rand.grid_scores_)
    print("===="*60)
    print(rand.best_params_)



#训练数据生成训练模型
def start_train(folder_name,count_vect,tfidf_transformer):
    train_data = datasets.load_files(folder_name)
    fea_train_counts = count_vect.fit_transform(train_data.data)
    train_tfidf = tfidf_transformer.fit_transform(fea_train_counts)

    #朴素贝叶斯 alpha=0.01
    clf = MultinomialNB(alpha=0.01).fit(train_tfidf, train_data.target)
    #opt_parameter(clf,train_data)

    #BernoulliNB
    #bnl_clf=BernoulliNB(alpha=0.01).fit(train_tfidf, train_data.target)

    #KNN
    #knnclf = KNeighborsClassifier()
    #knnclf.fit(train_tfidf, train_data.target)

    #RandomForestClassifier
    #rdft_clf=RandomForestClassifier()
    #rdft_clf.fit(train_tfidf, train_data.target)

    #SVC   参数 : C、kernel、degree、gamma、coef0
    #svclf = SVC(C=2,kernel='linear')
    #svclf.fit(train_tfidf, train_data.target)

    #GDBT  GradientBoostingClassifier n_estimators=200
    #gdbc_clf = GradientBoostingClassifier()
    #gdbc_clf .fit(train_tfidf, train_data.target)

    # 持久化存储模型
    #joblib.dump(clf, "naive_bayes.model")
    # 加载已存储的模型
    # clf = joblib.load("naive_bayes.model")
    return  clf


#通过训练模型来对测试数据进行预测
def start_predict(clf,test_data,count_vect,tfidf_transformer):

    fea_test_counts = count_vect.transform(test_data)
    test_tfidf = tfidf_transformer.transform(fea_test_counts)

    predicted = clf.predict(test_tfidf)
    print("===" * 60)

    #precision = metrics.accuracy_score(test_data.target, predicted)

    #之前对0值问题的粗暴处理
    #predicted= handle_zero_result(predicted)

    for item in predicted:
        print(item)
    '''
    print("准确率：")

    print(precision)
    '''
    #打印
    #for test_line, category in zip(test_data, predicted):
       # print(category)
        #print('%r => %s' % (test_line, test_data.target_names[category]))

    print("Size of fea_test:" + repr(test_tfidf.shape))
    return predicted





if __name__ == '__main__':

    #词频矩阵化类
    count_vect = CountVectorizer()
    #tfidf转换器
    tfidf_transformer = TfidfTransformer()

    # 获得测试文件的ID
    file = open("user_tag_query.2W.TEST", encoding="utf-8")
    test_data_id = []
    for line in file.readlines():
       # print(line.split()[0])
        test_data_id.append(line.split()[0])

    #获取age训练数据集
    #age_train_data = datasets.load_files("age_train_data_folder")
    #获取age测试数据集
    #age_test_data = datasets.load_files("age_test_data_folder")

    #获取处理好的测试数据集
    test_data = seg_test_data("user_tag_query.2W.TEST")

    #得到训练模型
    age_clf = start_train("age_train_data_folder", count_vect, tfidf_transformer)
    #开始预测
    age_predict = start_predict(age_clf,test_data, count_vect, tfidf_transformer)

    #education
    #education_train_data = datasets.load_files("education_train_data_folder")
    #education_test_data = datasets.load_files("education_test_data_folder")
    education_clf = start_train("education_train_data_folder", count_vect, tfidf_transformer)
    education_predict = start_predict(education_clf, test_data,count_vect,tfidf_transformer)

    # gender
    #gender_train_data = datasets.load_files("gender_train_data_folder")
    #gender_test_data = datasets.load_files("gender_test_data_folder")
    gender_clf = start_train("gender_train_data_folder", count_vect, tfidf_transformer)
    gender_predict = start_predict(gender_clf, test_data, count_vect, tfidf_transformer)


    #打印结果
    id = np.array(test_data_id)
    age = np.array(age_predict)
    gender = np.array(gender_predict)
    education = np.array(education_predict)
    #对矩阵转置
    id_ = id.transpose()
    age_ = age.transpose()
    gender_ = gender.transpose()
    education_ = education.transpose()
    #矩阵合并
    result = np.column_stack((id_,age_, gender_, education_))
    for item in result:
        print(item[0]+" "+item[1]+" "+item[2]+" "+item[3])

##TODO 编码格式
    #np.savetxt('result.csv', result, fmt='%s')



