# Depression_detection_dashboard
**Dashboard for depression detection through text.**
Deployed on EC2 server: http://13.233.89.242:8000

![Depression-check](/assets/Screenshot.png)

*Data used to train the model-*

As exclusive text data for a depression classification task was not readily available, I assembled a custom dataset from reddit posts and comments.
Posts from /r/depression and /r/suicidewatch were all labelled as 1. Posts and comments from other popular subreddits such as news, Askreddit, politics, etc. 
were labelled as 0. Some conditional preprocessing on data is done based on the label of the text to lower the chances of overfitting.

Total number of depression posts = 572841
Total number of non-depression posts = 817173

*Model-*

I have fine tuned the pretrained tensorflow roBERTa sequence classification model from huggingface library on the train data.
2 to 3 epochs is enough to gain an accuracy of 97-98% in validation and test sets. Beyond that, the accuracy falls due to loss of generalization.
Baseline methods like XGboost, passiveAggressive classifier give a validation,test set accuracy of 75-87%.

*Topic Modelling-*

Gensim's LDA mallet model extracts major topics in text marked as depression positive. Number of topics is tuned by maximizing coherence score(c_v) 
(more on that [here](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0)
).

*Web App-*

The web application is built using Dash, which uses Flask under the hood.
