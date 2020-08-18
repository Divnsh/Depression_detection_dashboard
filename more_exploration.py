#import re
#import numpy as np
import pandas as pd
from pprint import pprint
import os,shutil
# Gensim
import gensim
import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pickle
## Topic modelling - LDA

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

DATA_DIR = './Data'
PLOTS_DIR = './Plots'
MODELS_DIR = './Models'

def treat_for_lda(dat):
    dat=dat.apply(lambda x:x.split())
    data_lemmatized=list(dat)

    PLOTS_DIR='./Plots'
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    MODELS_DIR='./Models'
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])

    # Human readable format of corpus (term-frequency)
    print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    return corpus,data_lemmatized,id2word


# Build LDA model
# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=20,
#                                            random_state=2020,
#                                            update_every=2,
#                                            chunksize=1000,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
#
# # Print the Keyword in the topics
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]
#
# # Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
#
# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)
#
# # Visualize the topics
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis
# pyLDAvis.save_html(vis, os.path.join(PLOTS_DIR,'lda.html'))

##LDA Mallet Model

# Show Topics
# pprint(ldamallet.show_topics(formatted=False))
#
# # Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# the dominant topic in each sentence

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def save(lda):
    lda.save(os.path.join(MODELS_DIR,"lda.model"))

def load():
    lda = gensim.models.wrappers.LdaMallet.load(os.path.join(MODELS_DIR,"lda.model"))
    return lda


# dominant topics in input
def format_topics_inputs(input_data):
    # Init output
    sent_topics_df = pd.DataFrame()
    corpus, data_lemmatized, id2word = treat_for_lda(input_data['treated_lemmatized_text'])
    ldamodel=load()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j in [0,1,2]:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution']
    return(sent_topics_df)


if __name__=='__main__':
    assert os.path.exists(DATA_DIR), "Data path does not exit. Run scraping/cleaning modules first."
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_reddit.csv"))
    data = data.drop_duplicates().dropna()
    data = data.loc[data['label'] == 1, 'treated_lemmatized_text']

    corpus,data_lemmatized,id2word = treat_for_lda(data)

    os.environ['MALLET_HOME'] = './resources/mallet-2.0.8/'
    mallet_path = './resources/mallet-2.0.8/bin/mallet'
    #ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word, prefix=
    # os.path.join(MODELS_DIR,'lda_'))

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
                                                            start=2, limit=40, step=6)
    limit = 40;
    start = 2;
    step = 6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    if os.path.exists(os.path.join(MODELS_DIR,'lda_')):
        os.rmdir(os.path.join(MODELS_DIR,'lda_'))
    optimal_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=26, id2word=id2word,
                                                 prefix=os.path.join(MODELS_DIR,'lda_'))
    save(optimal_model)
    # Select the model and print the topics
    #optimal_model = model_list[4]
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))

    with open(os.path.join(MODELS_DIR, 'lda.model'), 'wb') as f:
        pickle.dump(optimal_model, f)
    model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)
    vis = pyLDAvis.gensim.prepare(model, corpus, id2word, n_jobs=4)
    pyLDAvis.show(vis)
    pyLDAvis.save_html(vis, os.path.join(PLOTS_DIR, 'lda.html'))

    # Below code for finding dominant topics in each text, etc....
    # assert os.path.exists(os.path.join(MODELS_DIR, 'lda.model')), "Topic model not saved. Please train the model first!"
    # with open(os.path.join(MODELS_DIR, 'lda.model'), 'rb') as f:
    #     optimal_model = pickle.load(f)
    #
    # df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
    #
    # # Format
    # df_dominant_topic = df_topic_sents_keywords.reset_index()
    # df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    #
    # # Show
    # df_dominant_topic.head(10)
    #
    # # the most representative document for each topic
    #
    # # Group top 5 sentences under each topic
    # sent_topics_sorteddf_mallet = pd.DataFrame()
    #
    # sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    #
    # for i, grp in sent_topics_outdf_grpd:
    #     sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
    #                                              grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
    #                                             axis=0)
    #
    # # Reset Index
    # sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
    #
    # # Format
    # sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    #
    # # Show
    # sent_topics_sorteddf_mallet.head()
    #
    # #  Topic distribution across documents
    #
    # # Number of Documents for Each Topic
    # topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    #
    # # Percentage of Documents for Each Topic
    # topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    #
    # # Topic Number and Keywords
    # topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    #
    # # Concatenate Column wise
    # df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    #
    # # Change Column names
    # df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    #
    # # Show
    # print(df_dominant_topics)
    # df_dominant_topics.to_csv('./resources/df_dominant_topics.csv',header=True,index=False)


