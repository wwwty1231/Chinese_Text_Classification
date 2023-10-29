import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

train_data = pd.read_csv('content/cnews.train.txt', sep='\t', names=['label', 'content'])
test_data = pd.read_csv('content/cnews.test.txt', sep='\t', names=['label', 'content'])

train_data.info()

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


if __name__ == "__main__":
    # #分词
    # train_content =train_data['content'].apply(chinese_word_cut)
    # test_content = test_data['content'].apply(chinese_word_cut)


    # 加载保存的jieba分词结果
    train_content = pd.read_csv('train_seg_content.txt', header=None, squeeze=True)
    test_content = pd.read_csv('test_seg_content.txt', header=None, squeeze=True)

    # 加载预训练的BERT模型和tokenizer
    model_name = 'bert-base-chinese'
    model_path = r'models\bert-base-chinese'

    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 将文本转换为BERT词向量
    def convert_text_to_bert_vectors(text):
        inputs = tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
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