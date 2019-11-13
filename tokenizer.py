#Homework 2, 4.1

import sys
import re
import string as s
from nltk.tokenize import word_tokenize

filepath = sys.argv[1] #filepath specified as second argument in the commandline
filetext = ''.join(open(filepath,'r').readlines()) #grab text as string

def simple_tokenizer(text):
    text = re.sub('\n',' ',text) #remove newline character
    text = re.sub('\'','',text) #remove apostrophes (e.g. so 'haven't -> havent')
    words = re.sub(r'[^\w\s]',' ',text) #replace all punctuation with spaces
    tokens = words.split(' ') #tokenise
    tokens = [tok for tok in tokens if tok != ''] #remove empty-strings
    return tokens

def off_shelf_tokenizer(text):
    tokens = word_tokenize(text) #nltk tokenizer
    return tokens


#SIMPLETOKENIZER
print('Simple tokenizer from scratch:')
print(simple_tokenizer(filetext))
print('\n')
print('NLTK tokenizer:')
print(off_shelf_tokenizer(filetext))
print('\n')
