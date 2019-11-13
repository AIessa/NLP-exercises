#Homework 2, 4.2

#imports -------------------------------------------------------------------#
import os
import re
import csv

#get desired document text for preprocessing & analysis --------------------#
  #all documents in "train" directory

directory_path = 'movies/train'
corpus_train = ""
for filename in os.listdir(directory_path):
    filepath = os.path.join(directory_path,filename)
    if os.path.isfile(filepath):
        text = ''.join(open(filepath,'r').readlines())
        corpus_train += text+' '

#preprocessing: tokenize ---------------------------------------------------#

def heavy_norm_tokenizer(text):
    text = re.sub('\n',' ',text) #remove newline character
    text = re.sub('\'',' \'',text) #remove apostrophes (e.g. so 'haven't -> havent')
    words = re.sub(r'[^\w\s\']',' ',text) #replace all punctuation with spaces
    normalized = words.lower()
    tokens = normalized.split(' ') #tokenise
    tokens = [tok for tok in tokens if tok != ''] #remove empty-strings
    return tokens

tokens = heavy_norm_tokenizer(corpus_train)

#analysis: count unigrams --------------------------------------------------#

def create_ngrams(n, tokens):
    ngram_list = []
    for tok_location, tok in enumerate(tokens):
        if tok_location <= len(tokens)-n: #stops at last possible index for ngram length
            ngram = ''
            for i in range(n): #n=3 -> [0,1,2]
                ngram += tokens[tok_location+i]
                ngram += ' '
            ngram_list.append(ngram)
    return ngram_list


def count_ngrams(all_tokens,unique_tokens):
    vocab_counts_over25 = []
    occurs_once = []
    occurs_twice = []
    occurs_thrice = []
    occurs_fourtimes = []
    for token in unique_tokens:
        tokcount = all_tokens.count(token)
        if tokcount >= 25:
            vocab_counts_over25.append((tokcount,token))
        elif tokcount ==1:
            occurs_once.append(token)
        elif tokcount ==2:
            occurs_twice.append(token)
        elif tokcount ==3:
            occurs_thrice.append(token)
        elif tokcount ==4:
            occurs_fourtimes.append(token)

    return [vocab_counts_over25,occurs_once,occurs_twice,occurs_thrice,occurs_fourtimes]


#count unique unigrams----------------------
print("Unique ngram counts 1,2,3:")
all_unigrams = create_ngrams(1,tokens)
unique_unigrams = list(set(all_unigrams))
print(len(unique_unigrams))

all_bigrams = create_ngrams(2,tokens)
unique_bigrams = list(set(all_bigrams))
print(len(unique_bigrams))

all_trigrams = create_ngrams(3,tokens)
unique_trigrams = list(set(all_trigrams))
print(len(unique_trigrams))

#vocabulary for unigrams--------------------
count_lists = count_ngrams(all_unigrams,unique_unigrams)

print("Occurs once: "+str(len(count_lists[1])))
print("Occurs twice: "+str(len(count_lists[2])))
print("Occurs thrice: "+str(len(count_lists[3])))
print("Occurs four times: "+str(len(count_lists[4])))

#get 10 most frequent--------------------------
vocab_counts_over25 = count_lists[0] #(tokcount,token)

def getkey(item):#helperfunction for sorting by key
    return item[0]
#sort
sorted_vocab = sorted(vocab_counts_over25, key = getkey, reverse = True)
print("Top 10 occurring unigrams:")
print(sorted_vocab[:10])

#print vocabulary with frequencies to csv file
with open("vocab25.csv", "w") as the_file:
    writer = csv.writer(the_file)
    for tup in vocab_counts_over25:
        writer.writerow(tup)
