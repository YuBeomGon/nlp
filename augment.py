#code is from https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py

import random
from random import shuffle
import re
# random.seed(1)

def get_only_chars(line):

    line = line.replace(',', '')
    line = line.replace("'", "")
    line = line.replace('-', '') 
    line = line.replace('\t', ' ')
    line = line.replace(':', '')
#     line = re.sub('[-:]+', '', line)
    line = line.lower()
    line = re.sub('[^가-힣A-Za-z0-9., ]+', '', line)
    line = re.sub('[0-9]+', ' ', line)
    line = re.sub('[ ]+', ' ', line)
    
    return line.strip()

def random_deletion(words, p):

    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

def random_swap(words, n):
    new_words = words.copy()
    if n == 0:
        return new_words
    
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

# def random_insertion(words, n):
#     new_words = words.copy()
#     for _ in range(n):
#         add_word(new_words)
#     return new_words

# def add_word(new_words):
#     synonyms = []
#     counter = 0
#     while len(synonyms) < 1:
#         random_word = new_words[random.randint(0, len(new_words)-1)]
#         synonyms = get_synonyms(random_word)
#         counter += 1
#         if counter >= 10:
#             return
#     random_synonym = synonyms[0]
#     random_idx = random.randint(0, len(new_words)-1)
#     new_words.insert(random_idx, random_synonym)
    
# def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):    
def text_aug(sentence, random_swap_p=0.02, random_del_p=0.05):    
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    swap_num = max(1, int(random_swap_p*num_words))  

    #rs
    a_words = random_swap(words, swap_num)
    #rd
    a_words = random_deletion(a_words, random_del_p)
    
    return ' '.join(a_words)