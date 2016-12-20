import jieba
import codecs
import jieba.analyse
import os

COUNT_TEST=0
COUNT_AGE = 0
COUNT_GENDER = 0
COUNT_EDUCATION = 0


#存储文件
def save(filename, contents):
  fh = codecs.open(filename, 'w', encoding='utf-8')
  fh.write(contents)
  fh.close()


#分类训练数据
#sample_content: 用户搜索的内容的分词
#type：用户的属性，0:ID,1:Age,2:Gender,3:Education
#value: 用户的各个属性的级别
def classify_data(sample_content, type, value):
    global COUNT_AGE, COUNT_GENDER, COUNT_EDUCATION
    temp = ""
    # 对一个用户的所有的搜索的句子进行分词，调用jieba的默认模式
    if ( sample_content.split()[type] == value ):
        for sentence in sample_content.split()[4:]:
            temp += (sentence+" ")

        #去停用词方法一：
        stop=[line.strip() for line in open("stop_words.txt","r",encoding="utf-8").readlines()]
        seg_list = jieba.cut(temp, cut_all=False)
        seg_content = " ".join(set(seg_list)-set(stop))
        #去停用词方法二：
        '''
        jieba.analyse.set_stop_words("stop_words.txt")
        seg_content =" ".join(jieba.analyse.extract_tags(temp, 230))
        '''

        #print(seg_content)
        if(type == 1):
          COUNT_AGE += 1
          folder = "age_train_data_folder/0"
          if not os.path.exists(folder):
              os.makedirs(folder)

          folder = "age_train_data_folder/"+value
          if not os.path.exists(folder):  ###判断文件是否存在，返回布尔值
            os.makedirs(folder)
          save(folder + "/file_" + str(COUNT_AGE) + ".txt", seg_content)
        elif(type == 2):
            COUNT_GENDER += 1
            folder = "gender_train_data_folder/0"
            if not os.path.exists(folder):
                os.makedirs(folder)

            folder = "gender_train_data_folder/"+value
            if not os.path.exists(folder):
                os.makedirs(folder)
            save(folder + "/file_" + str(COUNT_GENDER) + ".txt", seg_content)
        elif(type == 3):
            COUNT_EDUCATION += 1
            #建一个名字为0的文件夹
            folder = "education_train_data_folder/0"
            if not os.path.exists(folder):
                os.makedirs(folder)

            folder = "education_train_data_folder/"+value
            if not os.path.exists(folder):
                os.makedirs(folder)
            #保存到文件
            save(folder+"/file_"+str(COUNT_EDUCATION)+".txt", seg_content)


#分类测试数据
def classfy_test_data(sample_content):
    global COUNT_TEST
    temp = ""
    for sentence in sample_content.split()[1:]:
        temp += (sentence + " ")
    stop = [line.strip() for line in open("stop_words.txt", "r", encoding="utf-8").readlines()]

    seg_list = jieba.cut(temp, cut_all=False)
    seg_content = " ".join(set(seg_list) - set(stop))
    print(seg_content)
    COUNT_TEST += 1
    folder = "predict_test_data_folder/"
    save(folder + "test/file_" + str(COUNT_TEST) + ".txt", seg_content)




#读取文件
file = open("user_tag_query.2W.TRAIN", encoding="utf-8")
#file_test = open("user_tag_query.2W.TEST", encoding="utf-8")

for line in file.readlines():
    #依次分3次，1代表age 2代表gender，3代表education
    for type in range(1, 4):
      # 对于age，对0,1,2,3,4,5,6进行依次分类，放入建立好的文件夹中
      #classify_data(line, type, "0")
      classify_data(line, type, "1")
      classify_data(line, type, "2")
      classify_data(line, type, "3")
      classify_data(line, type, "4")
      classify_data(line, type, "5")
      classify_data(line, type, "6")
