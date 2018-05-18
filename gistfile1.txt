import gensim
from keras.preprocessing.text import text_to_word_sequence
import pymorphy2
from tqdm import tqdm

morph = pymorphy2.MorphAnalyzer()


necessary_part = {"NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND"}
with open('text.txt', 'r', encoding='utf8') as f:
    text = f.read().split('\n')
    res = []

    # Normalization
    for line in text:
        res.append(text_to_word_sequence(line))

    for i in tqdm(range(len(res))):
        res_lst = []
        for el in res[i]:
            p = morph.parse(el)[0]
            if p.tag.POS in necessary_part:
                res_lst.append(p.normal_form)
        res[i] = res_lst
    res = [x for x in res if x]
    model_name = 'my.model'

    # --------------------------
    # Training
    model = gensim.models.FastText(res, size=300, window=3, min_count=2, sg=1, iter=35)

    model.init_sims(replace=True)

    model.save(model_name)
    # --------------------------
    # Inference
    model = gensim.models.FastText.load(model_name)
    model.init_sims(replace=True)

    words = ['челавек',  'стулент', 'студечнеский', 'чиловенчость',
             'учавствовать', 'тактка', 'вообщем', 'симпотичный', 'зделать',  'сматреть', 'алгаритм', 'ложить']
    words_correct = ['человек', 'студент', 'студенческий', 'человечность',
                     'участвовать', 'тактика', 'вообще', 'симпатичный', 'сделать',
                     'смотреть', 'алгоритм', 'положить']

    for index, word in enumerate(words):
        if word in model:
            print(word)

            for i in model.most_similar(positive=[word], topn=20):
                print(i[0], i[1])
            print('---------------')
            print(model.similarity(word, words_correct[index]))
            print('______________')

            print('\n')
    # --------------------------
