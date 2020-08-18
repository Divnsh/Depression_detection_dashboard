import pandas as pd
import os
from wordcloud import STOPWORDS
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
#import plotly
#import chart_studio.plotly as py
#import plotly.tools as tls
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import emoji
import re
#import sys
#from collections import Counter
from contractions import contractions_dict
import spacy
os.system('python -m spacy download en_core_web_sm')
from nltk.corpus import stopwords
from symspellpy.symspellpy import SymSpell,Verbosity
import pkg_resources
from googletrans import Translator
#import langid
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
polyglot_logger.setLevel("ERROR")
from multiprocessing import Pool,Manager
#import concurrent.futures
from functools import partial
#import dask.dataframe as dd
import time
#from dask.distributed import Client
#import string
#from transformers import BertTokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

emojis = emoji.UNICODE_EMOJI
translator = Translator()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(bigram_path, term_index=0, count_index=2)
suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
s = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)

### Exploration ###

class Exploration():
    def __init__(self,posts):
        self.posts=posts
        self.N = 50
        self.DIDX = posts['label'] == 1

    def generate_ngrams(self,text, n_gram=1):
        token = [token for token in text.lower().split() if token != '' if token not in STOPWORDS]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [' '.join(ngram) for ngram in ngrams]

    def make_gram_df(self,n_gram=1):
        depression_grams = defaultdict(int)
        nondepression_grams = defaultdict(int)
        for t in self.posts[self.DIDX]['treated_lemmatized_text']:
            for word in self.generate_ngrams(t,n_gram):
                depression_grams[word] += 1
        for t in self.posts[~self.DIDX]['treated_lemmatized_text']:
            for word in self.generate_ngrams(t,n_gram):
                nondepression_grams[word] += 1
        dgramsdf = pd.DataFrame(sorted(depression_grams.items(), key=lambda x: x[1])[::-1])
        ndgramsdf = pd.DataFrame(sorted(nondepression_grams.items(), key=lambda x: x[1])[::-1])
        return dgramsdf,ndgramsdf

    def plot_grams(self,depression_grams,nondepression_grams,type):
        fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)
        plt.tight_layout()
        sns.barplot(y=depression_grams[0].values[:self.N], x=depression_grams[1].values[:self.N], ax=axes[0], color='red')
        sns.barplot(y=nondepression_grams[0].values[:self.N], x=nondepression_grams[1].values[:self.N], ax=axes[1], color='green')
        for i in range(2):
            axes[i].spines['right'].set_visible(False)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].tick_params(axis='x', labelsize=13)
            axes[i].tick_params(axis='y', labelsize=13)
        axes[0].set_title(f'Top {self.N} most common '+type+'grams in Depression posts', fontsize=15)
        axes[1].set_title(f'Top {self.N} most common '+type+'grams in Non-Depression comments', fontsize=15)
        #plotly_fig = tls.mpl_to_plotly(fig)
        # with open(os.path.join(PLOTS_DIR, 'plot_' + type+'.json'), 'w') as f:
        #     f.write(fig.to_json())
        #plotly.offline.plot(plotly_fig, filename=os.path.join(PLOTS_DIR, 'plot_' + type))
        plt.savefig(os.path.join(PLOTS_DIR, 'plot_' + type+'.png'))
        plt.show()

    def main(self):
        # Trigrams
        dtrigramsdf, ndtrigramsdf = self.make_gram_df(n_gram=3)
        # Bigrams
        dbigramsdf, ndbigramsdf = self.make_gram_df(n_gram=2)
        # Unigrams
        dunigramsdf, ndunigramsdf = self.make_gram_df(n_gram=1)
        self.plot_grams(dtrigramsdf,ndtrigramsdf,'tri')
        self.plot_grams(dbigramsdf,ndbigramsdf,'bi')
        self.plot_grams(dunigramsdf,ndunigramsdf,'uni')

### Cleaning ###

# Remove extra spaces, links and normalize
def remove_space(text):
    text = text.strip().lower()
    text = re.sub('http\S+', '', text)
    text = re.sub('www\S+', '', text)
    text = re.sub(r'[\n\t]', '',text)
    text = text.split()
    return " ".join(text)

def tokens_only(text):
    return ' '.join(word_tokenize(text))
    # return ' '.join(tokenizer.tokenize(text))

# Emoji handling

def emoji_handler(word):
    for emo in emojis.keys():
        if emo in word:
            word = word.replace(emo, " " + emojis[emo] + " ")
    return word

# Translate non-english words - takes too long (Still using it :face_with_tears_of_joy:)

def translatethis(word):
    try:
        if Detector(word).languages[0].code != 'en' and not bool(
                re.match(r'[a-z\./0-9\-…]+$', word)):
            if bool(re.search(r':[a-z_]+:', word)):
                spans = [(m.start(0), m.end(0)) for m in re.finditer(r':[a-z_]+:', word)]
                bord = word
                myspans = []
                for span in spans:
                    myspans.append(bord[span[0]:span[1]])
                    bord = bord.replace(bord[span[0]:span[1]], "")
                word = translator.translate(bord).text.lower() + " " + " ".join(myspans)
            else:
                word = translator.translate(word).text.lower()
    except:
        pass
    return word

# Spell correction

# def words(text):
#     return re.findall(r'\w+', text)
#
# WORDS = Counter(words(open(os.path.join(DATA_DIR,'big.txt'),encoding="utf8", errors='ignore').read()))
#
# def P(word, N=sum(WORDS.values())):
#     "Probability of `word`."
#     return WORDS[word] / N
#
# def correction(word):
#     "Most probable spelling correction for word."
#     return max(candidates(word), key=P)
#
# def candidates(word):
#     "Generate possible spelling corrections for word."
#     return (known([word]) or known(edits1(word)) or known(edits2(word)) or
#            [word])
#
# def known(words):
#     "The subset of `words` that appear in the dictionary of WORDS."
#     return set(w for w in words if w in WORDS)
#
# def edits1(word):
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits=[(word[:i],word[i:]) for i in range(len(word)+1)]
#     deletes =[L+R[1:] for L,R in splits if R]
#     transposes=[L+R[1]+R[0]+R[2:] for L,R in splits if len(R)>1]
#     replaces=[L+c+R[1:] for L,R in splits if R for c in letters]
#     inserts=[L+c+R for L,R in splits for c in letters]
#     return set(deletes+transposes+replaces+inserts)
#
# def edits2(word):
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correction2(wo):
    try:
        ans = sym_spell.lookup(wo, suggestion_verbosity, 2)[0].term
    except:
        ans = sym_spell.word_segmentation(wo).corrected_string
    return ans

def make_corrections(word):
    if not bool(re.match(r'[0-9\./,x\-\\\$£€₹¥\']+$', word)):
        nws = ''
        stripped_word = re.sub(r'\d+', '', word)
        if bool(re.search(':[a-z_]+:', stripped_word)):
            spans = [(m.start(0), m.end(0)) for m in re.finditer(r':[a-z_]+:', stripped_word)]
            bord = stripped_word
            myspans = []
            for span in spans:
                myspans.append(bord[span[0]:span[1]])
                bord = bord.replace(bord[span[0]:span[1]], "")
            for nw in bord.split():
                if len(nw) > 1:
                    nws += ' ' + correction2(nw)
                else:
                    nws += nw
            correct_maybe = nws + " " + " ".join(myspans)
            word = correct_maybe.strip()
        else:
            for nw in stripped_word.split():
                if len(nw) > 1:
                    nws += ' ' + correction2(nw)
                else:
                    nws += nw
            word = nws.strip()
    return word

# Contraction mapping
def mapping_replace(word):
    for w in contractions_dict.keys():
        if w in word:
            word = word.replace(w, " " + contractions_dict[w] + " ")
    word = ' '.join(word.split())
    return word

def mapping_contraction(text):
    for w in contractions_dict.keys():
        for word in text.split():
            if w in word:
                text = text.replace(w, " " + contractions_dict[w] + " ")
    return text

# Getting cleaned text
def replace_corrections(corpusdict, text):
    result = []
    for word in text.split():
        result.append(corpusdict[word])
    result = ' '.join(result)
    result = re.sub("\:", "", result)
    return result

# Stemming
def stemming(text):
    text = ' '.join(s.stem(word) for word in text.split())
    return text

# Lemmatization
def lemmatize(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# Stop words removal
def stopwords_treat(text):
    # Removing non ASCII chars
    # text = re.sub(r'[^\x00-\x7f]', r' ', text)
    # text = re.sub(r"\\", " ", text)
    # text = re.sub(r"[" + string.punctuation + "]+", " ", text)
    text = re.sub(r'[^a-z]', ' ', text)
    text = ' '.join([w for w in text.split() if not w in stop_words])
    return text

def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func)

def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

# Fitting distribution on normal post/comment lengths
# from fit_distribution import best_fit_distribution,make_pdf
# import scipy.stats as st
# data = posts[posts['label'] == 0].copy()['cleaned_text'].apply(lambda x: len(x.split(' . '))) # Text from after tokenization
# best_fit_name, best_fit_params = best_fit_distribution(data.values)
# best_dist = getattr(st, best_fit_name)
# pdf = make_pdf(best_dist, best_fit_params,size=1000000)
# pdf=pdf/sum(pdf)
# sizes=np.random.choice(pdf.index.values,size=750000,replace=True,p=pdf.values)
# np.save('./resources/sizes',sizes)

def treat_depression(cleaned):
    # Splitting depression texts into smaller segments for a better classification task
    text_to_treat=list(cleaned[cleaned['label'] == 1].copy()['cleaned_text'])[:int(.9*len(cleaned[cleaned['label'] == 1]))]
    text_untreated=list(cleaned[cleaned['label'] == 1].copy()['cleaned_text'])[int(.9*len(cleaned[cleaned['label'] == 1])):]
    mytext_to_treat = text_to_treat[:int(.8*len(text_to_treat))]
    rest_to_treat=text_to_treat[int(.8*len(text_to_treat)):]
    mytext_to_treat = [' '.join([w for w in x.split() if 'depress' not in w]) for x in mytext_to_treat] # Remove the word 'depress' from 72% of depressed posts
    text_to_treat=mytext_to_treat+rest_to_treat
    depressed_text = ' . '.join(text_to_treat)
    dsentences = depressed_text.split(' . ')
    dsentences = [s for s in dsentences if s!='' and '.' not in s]
    #sizes=np.clip(np.random.lognormal(3,2,750000),3,6000)
    try:
        sizes=np.load('./resources/sizes.npy')
    except:
        print("please save text sizes first in the resources directory - see fit_distribution.py")
    sizes=sizes+np.random.choice([1,0],len(sizes),p=[0.3,0.7])
    cumsize =0
    newtext=[]
    for s in sizes:
        if not dsentences[cumsize:int(cumsize+round(s,0))]:
            break
        newtext.append(" . ".join(dsentences[cumsize:int(cumsize+round(s,0))])+" . ")
        cumsize += int(s)
    t=list(cleaned[cleaned['label'] == 0].copy()['cleaned_text'])
    return pd.DataFrame({'cleaned_text':t+newtext+text_untreated, 'label':[0]*len(t)+[1]*len(newtext+text_untreated)})

def parallelize_on_rows_star(data,func,mydict,num_of_processes=8):
    pool=Pool(num_of_processes)
    result=pool.starmap(func,[(mydict,p.values[0]) for _,p in data[['cleaned_text']].iterrows()])
    pool.close()
    pool.join()
    return result

def update_mydict(func,dic,num_of_processes=8):
    # with concurrent.futures.ProcessPoolExecutor(8) as executor:
    #     k,v=zip(*dic.items())
    #     v=executor.map(func,v)
    #     return dict(zip(k,v))
    k, v = zip(*dic.items())
    pool = Pool(num_of_processes)
    v=pool.map(func,v)
    pool.close()
    pool.join()
    return dict(zip(k, v))

def Cleaning(posts,trainflag=1):
    start =time.time()
    print("Cleaning consists of 9 steps. Please be patient..")
    posts['cleaned_text'] = posts['Text'].apply(remove_space)
    print("Removed extra white spaces and links..")

    posts['cleaned_text']=parallelize_on_rows(posts['cleaned_text'],mapping_contraction)
    posts['cleaned_text'] = parallelize_on_rows(posts['cleaned_text'],tokens_only)
    # posts['cleaned_text'] = dd.from_pandas(posts, npartitions=8) \
    #     .map_partitions(lambda df: df.apply((lambda row: mapping_contraction(row['cleaned_text'])), axis=1),meta=('cleaned_text', 'object')).compute(scheduler=client) # Dask client causing semaphore leakages at times
    # posts['cleaned_text'] = dd.from_pandas(posts, npartitions=4*8) \
    #     .map_partitions(lambda df: df.apply((lambda row: tokens_only(row['cleaned_text'])), axis=1),meta=('cleaned_text', 'object')).compute(scheduler=client)
    if trainflag==1:
        posts=treat_depression(posts)
    corpus = ' '.join(list(posts['cleaned_text'])).split()
    corpus = list(set(corpus))
    corpusdict={}
    for w in corpus:
        corpusdict[w] = w
    print("Tokenization done..")

    corpusdict=update_mydict(emoji_handler, corpusdict)
    print("Emoji handling done..")

    corpusdict=update_mydict(translatethis, corpusdict)
    print("Translation of foreign words done..")

    corpusdict=update_mydict(make_corrections, corpusdict)
    print("Spelling corrections done..")

    corpusdict=update_mydict(mapping_replace, corpusdict)
    print("Contractions mapping done..")

    #corpusdict['vedio']='video'
    # posts['cleaned_text']= dd.from_pandas(posts, npartitions=8)\
    #     .map_partitions(lambda df: df.apply(lambda row: replace_corrections(corpusdict, row['cleaned_text']), axis=1),meta=('cleaned_text','object')).compute(scheduler=client)
    posts['cleaned_text'] = parallelize_on_rows_star(posts,replace_corrections,corpusdict)

    # posts['stemmed_text'] = dd.from_pandas(posts, npartitions=4*8) \
    #     .map_partitions(lambda df: df.apply(lambda row: stemming(row['cleaned_text']), axis=1),meta=('cleaned_text', 'object')).compute(scheduler=client)
    # posts['lemmatized_text'] = dd.from_pandas(posts, npartitions=4*8) \
    #     .map_partitions(lambda df: df.apply(lambda row: lemmatize(row['cleaned_text']), axis=1),meta=('cleaned_text', 'object')).compute(scheduler=client)
    posts['stemmed_text']=parallelize_on_rows(posts['cleaned_text'],stopwords_treat)
    posts['lemmatized_text']=parallelize_on_rows(posts['cleaned_text'],stopwords_treat)
    print("Stemming and lemmatization done..")

    posts['treated_stemmed_text'] = parallelize_on_rows(posts['stemmed_text'],stopwords_treat)
    posts['treated_lemmatized_text'] = parallelize_on_rows(posts['lemmatized_text'],stopwords_treat)
    # posts['treated_stemmed_text'] = dd.from_pandas(posts, npartitions=8) \
    #     .map_partitions(lambda df: df.apply(lambda row: stopwords_treat(row['cleaned_text']), axis=1),meta=('stemmed_text', 'object')).compute(scheduler=client)
    # posts['treated_lemmatized_text'] = dd.from_pandas(posts, npartitions=8) \
    #     .map_partitions(lambda df: df.apply(lambda row: stopwords_treat(row['cleaned_text']), axis=1),meta=('lemmatized_text', 'object')).compute(scheduler=client)
    print("Stop words treated..")

    end = time.time()
    print(f"Total time taken - {int((end - start)/60)} minutes,{(end - start)%60} seconds")
    return posts

if __name__=='__main__':
    DATA_DIR = './Data'
    postsdf = pd.read_csv(os.path.join(DATA_DIR, 'reddit_posts.csv'), encoding='utf-8')

    PLOTS_DIR = './Plots'
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    print('Data Shape = {}'.format(postsdf.shape))
    print('Memory Usage = {:.2f} MB'.format(postsdf.memory_usage().sum() / 1024 ** 2))

    #client = Client()
    cleaned=Cleaning(postsdf)
    cleaned=cleaned.drop_duplicates().dropna()
    cleaned.to_csv(os.path.join(DATA_DIR, "cleaned_reddit.csv"), header=True, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    print(f"File saved at: {os.path.join(DATA_DIR, 'cleaned_reddit.csv')}")
    exploration = Exploration(cleaned)
    exploration.main()

# sum(data[data['label']==0]['cleaned_text'].apply(lambda x: len(x)))/len(data[data['label']==0])
# data[data['label']==0]['cleaned_text'].apply(lambda x: len(x))
#
# sum(data[data['label']==0]['cleaned_text'].apply(lambda x: int(len(x) in range(300,500))))/len(data[data['label']==0])