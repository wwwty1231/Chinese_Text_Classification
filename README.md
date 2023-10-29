# 实验报告

## 环境

- Python 3
- TensorFlow 
- numpy
- scikit-learn
- transformers
- pandas

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

数据集划分如下：

- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

## 文本分类

#### 数据处理

##### 导入数据并将文字型的label 转为数字label

```python
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
# 加载训练数据和测试数据
train_data = pd.read_csv('content\cnews.train.txt', sep='\t', names=['label', 'content'])
test_data = pd.read_csv('content\cnews.test.txt', sep='\t', names=['label', 'content'])

def read_category(y_train):
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    label_id = []
    for i in range(len(y_train)):
        label_id.append(cat_to_id[y_train[i]])
    return label_id

train_target = train_data['label']
y_label = read_category(train_target)

test_target = test_data['label']
test_label = read_category(test_target)
```

##### 特征工程

**`jieba`分词**：使用 `jieba` 库对文本进行分词，然后根据预先定义的停用词列表 `stopwords.txt` 对分词结果进行过滤。

```python
def chinese_word_cut(mytext):
    # 使用jieba对中文文本进行分词处理
    seg_list = jieba.cut(mytext)
    seg_list_filtered = [word for word in seg_list if word not in stopwords]
    return " ".join(seg_list_filtered)


stopwords_path = 'stopwords.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]

# 分词处理
train_content = train_data['content'].apply(chinese_word_cut)
test_content = test_data['content'].apply(chinese_word_cut)
```

**特征提取**

​	TF-IDF是一种常用的文本特征表示方法，用于衡量一个词语在文档中的重要程度。

​	TF（词频）衡量了一个词在文档中的出现频率，它通过统计词语在文档中出现的次数来计算。TF可以简单地表示为一个词在文档中出现的次数，或者可以使用相对频率（出现次数除以文档中的词语总数）来表示。

​	IDF（逆文档频率）衡量了一个词语在整个文档集合中的稀有程度，即它在多少个文档中出现。IDF通过对整个文档集合中的文档数目进行除法运算的对数来计算。这个除法可以抵消词频的影响，使得常见词语的权重降低，罕见词语的权重提高。

TF-IDF的计算过程如下：

1. 分词：将文本分割成一个个词语或单词。

2. 计算TF：对于给定的文档，计算每个词语在文档中的词频，即该词在文档中出现的次数。

3. 计算IDF：对于文档集合中的每个词语，计算它的逆文档频率。逆文档频率的计算公式为：
   $$
   IDF = log(文档集合中的文档数目 / (包含该词语的文档数目 + 1))
   $$
   "+1"的目的是为了避免除以0的情况，平滑IDF的值。

4. 计算TF-IDF：将TF和IDF相乘，得到每个词语的TF-IDF权重。

手写实现如下：

```python
#加载自定义的停用词表
stopwords_path = 'stopwords.txt'	
	stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]   
# 构建词汇表
    vocab = set()
    for text in train_content:
        words = text.split()
        vocab.update(words)
    vocab = list(vocab)
    # 计算词语在文档中的TF-IDF值
    def calculate_tfidf(text):
        words = text.split()
        tfidf = {}
        for word in words:
            if word in tfidf:
                tfidf[word] += 1
            else:
                tfidf[word] = 1
        for word in tfidf:
            tfidf[word] *= math.log(len(train_content)/(sum(1 for text in train_content if word in text)+1))
        return tfidf
    # 提取训练数据和测试数据的特征
    X_train = []
    for text in train_content:
        tfidf = calculate_tfidf(text)
        features = [tfidf.get(word, 0) for word in vocab]
        X_train.append(features)

    X_test = []
    for text in test_content:
        tfidf = calculate_tfidf(text)
        features = [tfidf.get(word, 0) for word in vocab]
        X_test.append(features)

    train_target = train_data['label']
    test_target = test_data['label']
    y_label = read_category(train_target)
    test_label = read_category(test_target)
    
    data = (X_train, y_label, X_test, test_label)
    with open('data_tfi11df.pkl', 'wb') as fp:
        pickle.dump(data, fp)
```

#### 模型训练

##### 多项式朴素贝叶斯分类器

​	多项式朴素贝叶斯是一种朴素贝叶斯分类器的变种，通常用于处理文本分类问题。它基于贝叶斯定理，使用多项分布模型来估计文本中各个特征（词语）的条件概率，然后利用这些条件概率来进行分类。

1. 朴素贝叶斯假设：
多项式朴素贝叶斯基于朴素贝叶斯假设，即特征之间是条件独立的。即假设在文本分类问题中，每个特征(词语) 出现与其他特征的出现无关。

2. 贝叶斯定理：
多项式朴素贝叶斯使用贝叶斯定理来计算后验概率，即给定类别C，计算特征X的条件概率。这可以表示为：

​			$P(C|X)=\frac{P(X|C)\cdot P(C)}{P(X)}$ 

​			其中，$P(C|X)$ 是在给定特征X的条件下类别C的概率，$P(X|C)$ 是在类别C的条件下特征X的概率，$P(C)$ 是类别C的先验概率，		$P(X)$ 是特征X的先验概率。

3. 多项式模型：
   多项式朴素贝叶斯采用多项式分布模型来估计文本中各个特征的条件概率。在文本分类问题中，每个特征对应一个词语。假设文档中的词语服从多项式分布，多项式分布的参数表示每个词语在类别中出现的概率。

4. 参数估计：
   为了进行分类，需要估计以下参数：
   	$P(C)$: 类别的先验概率，通常通过训练数据中各个类别的文档数来估计。

​		   $P(X|C)$: 在类别C的条件下，特征X (词语) 的概率分布。



​	在这里我实现了多项式朴素贝叶斯分类器的`fit`和`predict`方法：

`fit` 方法：用于拟合模型，也就是根据训练数据计算类别的先验概率和特征的对数概率。具体步骤如下：

- 获取训练数据的样本数和特征数。
- 获取所有不同的类别。
- 初始化先验概率和特征的对数概率数组。
- 遍历每个类别，计算每个类别的先验概率和特征的对数概率。

`predict` 方法：这个方法用于预测新的数据点的类别。具体步骤如下：

- 计算数据点的联合对数似然性（`joint_log_likelihood`）。这涉及将输入数据点乘以特征的对数概率并加上类别的对数先验概率。
- 选择具有最大联合对数似然性的类别作为预测结果。这通常是通过 `argmax` 函数来实现的。

手写代码实现如下：

```python
import pickle
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.class_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            X_c = X_c.toarray()  # 转换稀疏矩阵为NumPy数组
            self.class_prior_[i] = (len(X_c) + self.alpha) / (n_samples + self.alpha * n_classes)
            self.feature_log_prob_[i] = np.log((X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * n_features))

    def predict(self, X):
        joint_log_likelihood = X @ self.feature_log_prob_.T + np.log(self.class_prior_)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
```

训练模型并进行预测

```python
# 加载数据
with open('data_tfidf.pkl', 'rb') as file:
    data = pickle.load(file)
X_train, y_label, X_test, test_label = data
# 创建并训练贝叶斯分类器
nb_classifier = MultinomialNaiveBayes()
nb_classifier.fit(X_train, y_label)
# 预测测试数据
y_pred = nb_classifier.predict(X_test)
# 计算准确率
accuracy = np.mean(y_pred == test_label)
print("准确率:", accuracy)
report = classification_report(test_label, y_pred)
print(report)
```

输出

```python
准确率: 0.908
              precision    recall  f1-score   support
           0       1.00      1.00      1.00      1000
           1       0.95      0.99      0.97      1000
           2       0.62      0.93      0.75      1000
           3       0.98      0.39      0.56      1000
           4       0.92      0.94      0.93      1000
           5       0.95      0.99      0.97      1000
           6       0.98      0.97      0.98      1000
           7       0.96      0.91      0.93      1000
           8       0.97      0.97      0.97      1000
           9       0.94      0.99      0.96      1000

    accuracy                           0.91     10000
   macro avg       0.93      0.91      0.90     10000
weighted avg       0.93      0.91      0.90     10000
```

#### 优化

​	为了优化性能我做出了下面的几个尝试。

##### 参数调优

​	网格搜索（Grid Search）：网格搜索是一种常用的参数调优方法，它通过在预定义的参数网格中穷举所有可能的参数组合来找到最佳参数。对于每个参数组合，使用交叉验证评估模型性能，并选择具有最佳性能的参数组合。

这里实现了使用网格搜索对多项式朴素贝叶斯分类器进行参数调优和交叉验证。

```python
import pickle

# 打开二进制文件以加载数据
with open('data_tfidf.pkl', 'rb') as file:
    data = pickle.load(file)

# 从加载的数据中获取训练集、标签和测试集
X_train, y_label, X_test, test_label = data

from sklearn.model_selection import GridSearchCV
from MultionmialNB import MultinomialNB

# 定义参数网格
param_grid = {'alpha': [0.001, 0.1, 0.2,0,5,0.7,1]}

# 创建多项式朴素贝叶斯分类器
nb_classifier = MultinomialNB()

# 创建GridSearchCV对象
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5)

# 执行参数调整和交叉验证
grid_search.fit(X_train, y_label)

# 输出最佳参数和最佳性能指标
print("最佳参数：", grid_search.best_params_)
print("最佳准确率：", grid_search.best_score_)
```

​	得到在`param_grid = {'alpha': [0.001, 0.1, 0.2,0,5,0.7,1]}`范围内的最佳参数： `{'alpha': 0.1}`，最佳准确率： `0.9293600000000002`。若要进一步优化则需缩小范围进行测试。

##### **高斯贝叶斯分类器**

高斯贝叶斯分类器与多项式贝叶斯的区别是，前者没有假定数据各维的独立性，使用多维高斯分布刻画数据的分布

$$
\begin{aligned}f(x)&=\frac{1}{\sqrt{\left(2\pi\right)^{n}\det\left(\Sigma\right)}}\exp\bigl(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\bigr)\end{aligned}
$$

$n$为数据维度，$\Sigma$为协方差矩阵 (含义是数据各维度的相关性), $\mu$为均值向量 (有$K$个)。高斯贝叶斯分类器还分共享协方差矩阵和不共享，如果共享，只有1个协方差矩阵，否则为 $K$ 个。

​	**特点：特征服从正态分布**

​	在这里我尝试实现了高斯贝叶斯分类器的`fit`和`predict`方法。

手写代码实现如下：

```python
class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}
        # 将稀疏矩阵转换为密集数组
        if isinstance(X, csr_matrix):
            X = X.toarray()
        # 计算每个类别的先验概率
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / len(y)
        # 计算每个类别的均值和方差
        for c in self.classes:
            X_c = X[y == c]
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_variances[c] = np.var(X_c, axis=0)

    def predict(self, X):
        predictions = []
        # 将稀疏矩阵转换为密集数组
        if isinstance(X, csr_matrix):
            X = X.toarray()
        for x in X:
            class_scores = []
            # 计算每个类别的后验概率得分
            for c in self.classes:
                prior = self.class_priors[c]
                mean = self.class_means[c]
                variance = self.class_variances[c]
                # 使用高斯分布的概率密度函数计算后验概率得分
                score = -0.5 * np.sum(np.log(2 * np.pi * variance))
                score -= 0.5 * np.sum(((x - mean) ** 2) / variance)
                score += np.log(prior)
                class_scores.append(score)
            # 选择具有最高后验概率得分的类别作为预测结果
            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)
        return predictions
```

PS:将稀疏矩阵转换为密集数组比较内存

训练模型并进行预测

```python
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
```

输出结果：

```python
准确率: 0.8486
				precision    recall  f1-score   support
           0       1.00      0.96      0.98      1000
           1       0.93      0.98      0.96      1000
           2       0.55      0.84      0.67      1000
           3       0.81      0.22      0.35      1000
           4       0.80      0.82      0.81      1000
           5       0.91      0.89      0.90      1000
           6       0.99      0.91      0.95      1000
           7       0.83      0.93      0.88      1000
           8       0.86      0.97      0.91      1000
           9       0.95      0.96      0.95      1000

    accuracy                           0.85     10000
   macro avg       0.86      0.85      0.84     10000
weighted avg       0.86      0.85      0.84     10000
```

发现准确率不如多项式朴素贝叶斯分类，分析可能原因：

​	**数据类型不匹配**：高斯朴素贝叶斯假设特征数据是**连续型**的，而多项式朴素贝叶斯适用于**离散型**数据。我使用**`tfidf`提取的特征可能接近多项分布而不是高斯分布**，高斯朴素贝叶斯可能无法有效地捕捉特征之间的关系。

##### 优化特征工程

​	于是乎想到了在特征提取上优化，因为曾经用过bert处理NLP任务，所以这次尝试了下用预训练的bert提取词向量。（PS:用bert提取特征需要大量的计算资源和内存，我放在服务器上跑了2个多小时）

```python
	# 加载预训练的BERT模型和tokenizer
    model_name = 'bert-base-chinese'
    model_path = r'models\bert-base-chinese'

    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 将文本转换为BERT词向量
    def convert_text_to_bert_vectors(text):
        inputs = tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # 获取CLS向量作为句子的表示
        return embeddings
    # 生成训练集和测试集的BERT词向量
    train_content = train_data['content'].tolist()
    test_content = test_data['content'].tolist()
    
    train_vectors = [convert_text_to_bert_vectors(text) for text in train_content]
    test_vectors = [convert_text_to_bert_vectors(text) for text in test_content]
    # 将BERT词向量转换为numpy数组
    X_train = torch.stack(train_vectors).numpy()
    X_test = torch.stack(test_vectors).numpy()
    
    import pickle
    data = (X_train, y_label, X_test, test_label)
    with open('data_bert.pkl', 'wb') as fp:
        pickle.dump(data, fp)
```

但是，用之前定义的`MultinomialNaiveBayes`和`GaussianNB`去进行分类，但发现准确率不如TF-TDF。分析原因：

​	贝叶斯分类器的模型结构与BERT提取的词向量不完全匹配，为适应其模型将BERT降维损失了一定信息，故准确率不高。

​	BERT提取的词向量是高维度的连续向量，包含了丰富的语义和上下文信息。在贝叶斯分类器中，如果特征维度过高，可能导致模型过于复杂和容易过拟合。而TF-IDF是一种稀疏的离散特征表示方法，对于贝叶斯分类器更容易处理。

##### SVM

​	于是尝试了下用SVM处理BERT提取的特征，时间原因这里就没有像多项式贝叶斯和高斯贝叶斯那里手写实现，简单的调库得出结果。

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
with open(r'data_bert.pkl', 'rb') as fp:
    data = pickle.load(fp)
    
X_train, y_train, X_test, y_test = data
X_train = np.mean(X_train, axis=1)
X_test = np.mean(X_test, axis=1)
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
report = classification_report(y_test, y_pred)
print(report)
```

输出

```python
准确率: 0.951
				precision    recall  f1-score   support
           0       1.00      0.99      1.00      1000
           1       0.94      0.98      0.96      1000
           2       0.83      0.90      0.87      1000
           3       0.97      0.79      0.87      1000
           4       0.97      0.94      0.95      1000
           5       0.95      0.98      0.96      1000
           6       0.95      0.98      0.97      1000
           7       0.95      0.96      0.95      1000
           8       0.98      0.99      0.99      1000
           9       0.99      0.98      0.98      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000
```

发现结果明显优于贝叶斯分类算法，分析可能原因:

​	SVM是一种非线性分类器，它可以通过使用核函数将数据从原始特征空间映射到高维特征空间，从而更好地处理非线性分类问题。BERT提取的词向量具有较高的维度，包含了丰富的语义和上下文信息，适合在高维空间中进行分类，而SVM能够更好地利用这些特征。

