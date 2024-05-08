import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

w2v_file_path = "word2vec_model.bin"
W2V_MODEL = Word2Vec.load(w2v_file_path)
W2V_SIZE = 100  # 词向量的维度。


def get_w2v(word):
    # TODO
    return W2V_MODEL.wv[word] if word in W2V_MODEL.wv.index_to_key else None
    # .wv属性提供了对模型词汇表和词向量的访问。它是一个字典-like对象，其中键是词汇表中的单词，值是对应的词向量。
    # word in W2V_MODEL.wv.index_to_key: 这个条件检查word是否存在于模型的词汇表中。index_to_key是Word2Vec模型的一个属性，它提供了从索引到词汇表中单词的映射。如果word在模型的词汇表中，那么这个条件会返回True。


def get_sentence_vector(sentence):
    # TODO
    vec = []
    for word in sentence:
        word_vec = get_w2v(word)
        if word_vec is not None:
            vec.append(word_vec)
    return np.mean(vec, axis=0) if len(vec) != 0 else np.zeros(W2V_SIZE)


def get_similarity(array1, array2):
    array1_2d = np.reshape(array1, (1, -1)) # -1表示自动推断缺失的维度大小
    # 这样做是因为cosine_similarity函数期望二维数组作为输入。
    array2_2d = np.reshape(array2, (1, -1))
    similarity = cosine_similarity(array1_2d, array2_2d)[0][0]
    # cosine_similarity函数返回的是一个二维数组，其中每个元素表示一对向量的相似度。由于我们只比较两个向量，所以返回的结果是一个单元素的矩阵。因此，我们使用[0][0]来提取并返回余弦相似度的值。
    return similarity


def main():

    # 测试两个句子
    sentence1 = "我不喜欢看新闻。"
    sentence2 = "我觉得新闻不好看。"
    sentence_split1 = jieba.lcut(sentence1)
    sentence_split2 = jieba.lcut(sentence2)
    # 获取句子的句向量
    sentence1_vector = get_sentence_vector(sentence_split1)
    sentence2_vector = get_sentence_vector(sentence_split2)
    # 计算句子的相似度
    similarity = get_similarity(sentence1_vector, sentence2_vector)
    print(similarity)


if __name__ == "__main__":
    main()
