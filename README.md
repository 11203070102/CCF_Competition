# CCF_Competition

1，环境配置：

   机器学习语言和库分别使用的是Python3.5+sklearn
  环境搭建使用的Jetbrains公司开发的Pychram 社区版(使用学校邮箱注册可以免费使用Jetbrains公司的所有的产品）和
  Anaconda(主要是单独下载sklearn的那一系列的库，很难成功，不同的库之间有相互的关联，使用Pychram是很难一次安装成功
  所以直接使用Andconda可以一次性将所有与机器学习相关的100多个库安装进来)


2， 文件介绍：
   先运行classify_train_data_utils对数据分类，然后再运行main
   classify_train_data_utils.py  对训练集进行分词，并存入文件
   jieba-master 使用的结巴分词库来进行对文本的分词
   main.py 主函数
   stop_words.txt  停用词集合，用于去停用词

   age_test_data_folder(这里没有)   age_train_data_folder  这两类文件分别为测试文件集和训练文件集，是从20000条训练数据user_tag_query.2W.TRAIN中划分的。其中user_tag_query.5.TEST为测试文件5000条，训练文件集user_tag_query.15.TRAIN为15000条，通过这样分可以在本地看到准确率，和调参。通过调好的参数和模型，最终通过对user_tag_query.2W.TEST进行预测获取预测结果。划分文件集合很重要，这里面包含着数据的标签。当然这里我的代码里没有了age_test_data_folder等文件，主要是这个版本的代码编写里面，为简单起见没有如上所说，划分训练数据使用本地测试，而是直接使用的2W条训练数据进行训练，然后使用2W条测试数据进行预测，有兴趣可以自己划分训练数据集测试模型的预测效果。


