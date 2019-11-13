#training language (unigram/bigram) models for positive and negative articles

#imports -----------------------------------------------------------------------
import re
import os
import math
from nltk.corpus import stopwords

# create separate corpus/vocab with Pos/Neg articles ---------------------------
corpus_pos_train = ""
corpus_neg_train = ""

#counters for documents in each class & total
N_pos = 0
N_neg = 0
N_doc = 0

for filename in os.listdir('movies/train'):
    filepath = os.path.join('movies/train',filename)
    if os.path.isfile(filepath):
        N_doc += 1
        #if positive, add to positive
        if re.match(r'P-[a-z]*[0-9]*.txt',filename):
            text = ' '.join(open(filepath,'r').readlines())
            corpus_pos_train += '<S> '+text+' ' #add "<S>" to indicate start of article
            N_pos += 1
        #if negative, add to negative
        elif re.match(r'N-[a-z]*[0-9]*.txt',filename):
            text = ' '.join(open(filepath,'r').readlines())
            corpus_neg_train += '<S> '+text+' ' #add "<S>" to indicate start of article
            N_neg += 1

# preprocessing functions ------------------------------------------------------

#function: tokenize
def heavy_norm_tokenizer(text):
    text = re.sub('\n',' ',text) #remove newline character
    text = re.sub('\'',' \'',text) #remove apostrophes (e.g. so 'haven't -> havent')
    words = re.sub(r'[^\w\s\'\<S>]',' ',text) #replace all punctuation with spaces (leave apostrophes and <S> alone)
    normalized = words.lower()
    tokens = normalized.split(' ') #tokenise
    tokens = [tok for tok in tokens if tok != ''] #remove empty-strings
    #remove stopwords
    #stop = set(stopwords.words('english'))
    #tokens = [token for token in tokens if (not (token in stop))]
    return tokens

#function: create n-gram with given n
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

def create_bigrams(n,tokens): #for different bigram format (tuple not string)
    ngram_list = []
    for tok_location, tok in enumerate(tokens):
        if tok_location <= len(tokens)-n: #stops at last possible index for ngram length
            ngram = []
            for i in range(n): #n=3 -> [0,1,2]
                ngram.append(tokens[tok_location+i])
            ngram_list.append(ngram)
    return ngram_list


# unigram language model -------------------------------------------------------


#training---------

def train_model(bigrams, unigrams, Vtot, N_class, N_doc, k):
    #P(c) = logprior is log(docs in class c/tot docs) -> MLE!
    logprior = math.log((float(N_class)/float(N_doc)),2)

    #V = Vtot -> total vocabulary disregarding classes
    #bigdoc[c] = unigrams -> all unigrams from docs in class c
    #smoothing factor k to avoid negative probabilities

    print(len(Vtot))
    track_processing_count = 0

    loglikelihood = {} #this will be for each follow-word using bigram info
    for bigram in Vtot: #all (unique) bigrams: (wi-1,wi)
        w_prev = bigram[0]
        word = bigram[1]
        #check if word already in loglikelihood: means that all possible bigrams ending with it have been added!
        #if yes, move on. Else add it!
        if not(word in loglikelihood):
            print('preceeding \"'+word+'\":')
            #for w: get all possible w_prev & likelihoods
            wprevs_for_w = [wprev for (wprev,w) in Vtot if (w == word)]
            w_prev_dict = {}
            for wprev in wprevs_for_w:
                print('\t'+wprev)
                count_tuple = float(bigrams.count([wprev,word]))
                count_wprev = float(unigrams.count(wprev))
                w_prev_dict[wprev] = math.log(((count_tuple+k)/(count_wprev+(k*len(Vtot)))),2) #logLIKELIHOOD of wprev|word #
            loglikelihood[word] = w_prev_dict
            #look up any wprev|word with: loglikelihood[word][wprev]
    return logprior , loglikelihood


#testing -----------

#test does NOT compare classes (i.e. lang models for each class), just returns probabilities for one
def test_model(testdoc_tokens, model, Vtot):
    prob_modelclass = model[0] #add logprior of class from trained model
    for bigram in testdoc_tokens:
        word = bigram[1]
        wprev = bigram[0]
        if bigram in Vtot: #ignore if not in trained vocab
            prob_modelclass += model[1][word][wprev] #add loglikelihood for word from trained model
    return prob_modelclass

#this DOES give most likely class based on both models
def test_getclass(testdoc_tokens,model_pos,model_neg,Vtot):
    prob_pos = test_model(testdoc_tokens, model_pos, Vtot)
    print('Pos with: '+str(prob_pos))
    prob_neg = test_model(testdoc_tokens, model_neg,Vtot)
    print('Neg with: '+str(prob_neg))
    if prob_pos > prob_neg:
        print("Classification: P")
        return "P"
    elif prob_neg > prob_pos:
        print("Classification: N")
        return "N"
    else:
        print("Classes are equally probable")
        return "?"



# actually train models (training phase) ------------------------------------------------

#set input parameters:
print("Tokenizing and preparing bigrams ...")
#tokens per class
pos_unigrams = create_ngrams(1,heavy_norm_tokenizer(corpus_pos_train))
neg_unigrams = create_ngrams(1,heavy_norm_tokenizer(corpus_neg_train))
pos_bigrams = create_bigrams(2,heavy_norm_tokenizer(corpus_pos_train))
neg_bigrams = create_bigrams(2,heavy_norm_tokenizer(corpus_neg_train))
#whole vocabulary (unique tokens)
print("Setting parameters ...")
#Vtot = list(set(pos_bigrams + neg_bigrams))
Vtot = [list(x) for x in set(tuple(x) for x in (pos_bigrams+neg_bigrams))]
#smoothing factor
k = 1
print("Training models... k is set to "+str(k))
#train
pos_model = train_model(pos_bigrams, pos_unigrams , Vtot, N_pos, N_doc, k)
neg_model = train_model(neg_bigrams, neg_unigrams , Vtot, N_neg, N_doc, k)
print("Models trained!")


# test models ------------------------------------------------------------------
print("Testing models, classifying test data...\n")

correct_classification = 0
false_classification = 0

#open output document:
#f = open("output6_k10.txt","w+")

#get results for each test file (iterate through test directory):
for filename in os.listdir('movies/test'):
    filepath = os.path.join('movies/test',filename)
    if os.path.isfile(filepath):
        print(filename)
        #get text
        test_text = ' '.join(open(filepath,'r').readlines())
        #normalize/tokenize/n-gram-ize in same way as training data
        test_bigrams = create_bigrams(2,heavy_norm_tokenizer(test_text))
        #use models to predict category!
        c = test_getclass(test_bigrams,pos_model,neg_model,Vtot)
        print('\n')
        if c in filename:
            correct_classification +=1
        else:
            false_classification +=1
        #Append new info to output file
        #f = open("output6_k10.txt","a+")
        #f.write(filename[:-4]+'\t'+c+'\n')

print("Correctly classified: "+str(correct_classification))
print("Falsely classified: "+str(false_classification))
accuracy = (float(correct_classification)/(correct_classification+false_classification))*100
print("Accuracy: "+str(accuracy)+"%")
