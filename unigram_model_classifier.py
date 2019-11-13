#training language (unigram/bigram) models for positive and negative articles

#imports -----------------------------------------------------------------------
import re
import os
import math


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

# unigram language model -------------------------------------------------------


#training---------

def train_model(unigrams, Vtot, N_class, N_doc, k):
    #P(c) = logprior is log(docs in class c/tot docs) -> MLE!
    logprior = math.log((float(N_class)/float(N_doc)),2)

    #V = Vtot -> total vocabulary disregarding classes
    #bigdoc[c] = unigrams -> all unigrams from docs in class c
    #smoothing factor k to avoid negative probabilities

    loglikelihoods = {}
    for word in Vtot: #calculate P(w|c) terms
        count_w_c = float(unigrams.count(word)) #occurrences of word in bigdoc[c]
        len_class = float(len(unigrams)) #number of words in class
        len_Vtot = float(len(Vtot))
        likelihood_w_c = (count_w_c + k)/(len_class+(k*len_Vtot))
        loglikelihood_w_c = math.log(likelihood_w_c,2)

        loglikelihoods[word] = loglikelihood_w_c

    return logprior , loglikelihoods


#testing -----------

#test does NOT compare classes (i.e. lang models for each class), just returns probabilities for one
def test_model(testdoc_tokens, model, Vtot):
    prob_modelclass = model[0] #add logprior of class from trained model
    for word in testdoc_tokens:
        if word in Vtot: #ignore if not in trained vocab
            prob_modelclass += model[1][word] #add loglikelihood for word from trained model
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
print("Tokenizing and preparing unigrams ...")
#tokens per class
pos_unigrams = create_ngrams(1,heavy_norm_tokenizer(corpus_pos_train))
neg_unigrams = create_ngrams(1,heavy_norm_tokenizer(corpus_neg_train))

#whole vocabulary (unique tokens)
print("Setting parameters ...\nThis takes a while, because we are removing all words appearing less than 25 times in the whole collection in a really inefficient way! Please be patient...")
Vtot = list(set(pos_unigrams + neg_unigrams))

#######To remove all unigrams that appear less than 25 times:
Vtot = [uni for uni in Vtot if (pos_unigrams+neg_unigrams).count(uni) >= 25]
print('vocab: '+str(len(Vtot)))

pos_unigrams = [uni for uni in pos_unigrams if (uni in Vtot)]
neg_unigrams = [uni for uni in neg_unigrams if (uni in Vtot)]
print('positive unigrams: '+str(len(pos_unigrams)))
print('negative unigrams: '+str(len(neg_unigrams)))

#smoothing factor
k = 10
print("Training models...k is set to "+str(k))
#train
pos_model = train_model(pos_unigrams, Vtot, N_pos, N_doc, k)
neg_model = train_model(neg_unigrams, Vtot, N_neg, N_doc, k)
print("Models trained!")


# test models ------------------------------------------------------------------
print("Testing models, classifying test data...\n")

correct_classification = 0
false_classification = 0

#open output document:
f = open("output5_k10.txt","w+")


#get results for each test file (iterate through test directory):
for filename in os.listdir('movies/test'):
    filepath = os.path.join('movies/test',filename)
    if os.path.isfile(filepath):
        print(filename)
        #get text
        test_text = ' '.join(open(filepath,'r').readlines())
        #normalize/tokenize/n-gram-ize in same way as training data
        test_unigrams = create_ngrams(1,heavy_norm_tokenizer(test_text))
        #use models to predict category!
        c = test_getclass(test_unigrams,pos_model,neg_model,Vtot)
        print('\n')
        if c in filename:
            correct_classification +=1
        else:
            false_classification +=1
        #Append new info to output file
        f = open("output5_k10.txt","a+")
        f.write(filename[:-4]+'\t'+c+'\n')

f.close()

print("Correctly classified: "+str(correct_classification))
print("Falsely classified: "+str(false_classification))
accuracy = (float(correct_classification)/(correct_classification+false_classification))*100
print("Accuracy: "+str(accuracy)+"%")
