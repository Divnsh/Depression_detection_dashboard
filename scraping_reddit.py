import praw
import pandas as pd
from datetime import datetime
import urllib.request
import json
import requests
import time
import math
import numpy as np
from tqdm import tqdm
import os
import pickle
import random
import csv
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from multiprocessing import Pool

DATA_DIR='./Data'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# Initialize scraper
reddit = praw.Reddit(client_id ='kubhdgi6Cy5vaQ',
                     client_secret ='h28n4Gvh3Ksgzhpk6QMc8c6_u3w',
                     user_agent ='redscrape')

print(reddit.read_only) # check if read only instance

depression_subs = ['depression','SuicideWatch']
positive_subs = ['funny','MachineLearning','gaming','science','Positivity','todayilearned']
comment_subs = ['news','politics','AskReddit']
depression_subs_startdates=['2018-01-05','2018-01-05']
positive_subs_startdates=['2019-10-30','2019-10-30','2019-10-30','2019-10-30','2019-10-30']
comment_subs_startdates=['2019-10-30','2019-10-30','2019-10-30']
depression_subs_enddates=['2020-07-05','2020-07-05']
positive_subs_enddates=['2019-12-31','2019-12-31','2019-12-31','2019-12-31','2019-12-31']
comment_subs_enddates=['2019-12-31','2019-12-31','2019-12-31']

# Scraping posts
def make_request(uri, max_retries = 5):
    def fire_away(uri):
        response = requests.get(uri)
        assert response.status_code == 200
        #print(uri)
        return json.loads(response.content)
    current_tries = 1
    while current_tries < max_retries:
        try:
            time.sleep(1)
            response = fire_away(uri)
            return response
        except:
            time.sleep(1)
            current_tries += 1
    return fire_away(uri)


def pull_posts_for(subreddit, start_at, end_at):
    def map_posts(posts):
        for post in posts:
            try:
                post['selftext'] = post['selftext']
            except:
                post['selftext'] = ' '
        return list(map(lambda post: {
            'id': post['id'],
            'created_utc': post['created_utc'],
            'prefix': 't4_',
            'text': post['title']+ ' ' +post['selftext'],
        }, posts))
    SIZE = 100
    URI_TEMPLATE = r'https://api.pushshift.io/reddit/submission/search/?after={}&before={}&subreddit={}&size={}'
    post_collections = map_posts( \
        make_request( \
            URI_TEMPLATE.format( \
                start_at, end_at, subreddit, SIZE))['data'])
    n = len(post_collections)
    while n == SIZE:
        last = post_collections[-1]
        new_start_at = last['created_utc'] - (5)
        more_posts = map_posts( \
            make_request( \
                URI_TEMPLATE.format( \
                    new_start_at, end_at, subreddit, SIZE))['data'])
        n = len(more_posts)
        post_collections.extend(more_posts)
    return post_collections

def parellelize(allsubs,allstarts,allends):
    pool = Pool(8)
    result=pool.starmap(pull_posts_for,[(s, e, l) for s,e,l in zip(allsubs,allstarts,allends)])
    pool.close()
    pool.join()
    return result

# Removing duplicates
def remove_duplicates(posts):
    dpostids=[]
    dposts1=[]
    for post in posts:
        if post['id'] not in dpostids:
            if len(post['text'].split())>2:
                dposts1.append(post['text'])
        dpostids.append(post['id'])
    posts=dposts1
    print(f"Total depression submissions: {len(dposts)}")
    return dposts

# Extracting comments for comment_subs non-depression submissions
TIMEOUT_AFTER_COMMENT_IN_SECS = 0.150

def get_comments(red, submission_id, queue):
    submission = red.submission(id=submission_id)
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        if len(comment.body.split()) > 3:
            queue.put(comment.body)
        if TIMEOUT_AFTER_COMMENT_IN_SECS > 0:
            time.sleep(TIMEOUT_AFTER_COMMENT_IN_SECS)

def fetch_parallel(subids,idlen):
    result = queue.Queue()
    with ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(get_comments,[reddit]*idlen,subids,[result]*idlen)
        executor.shutdown(wait=True)
    return result

if __name__=='__main__':
    # All subreddit names and start-end dates to scrape
    allsubs = depression_subs + positive_subs + comment_subs
    allstarts = depression_subs_startdates + positive_subs_startdates + comment_subs_startdates
    allends = depression_subs_enddates + positive_subs_enddates + comment_subs_enddates
    nsubs = len(allsubs)

    # Scrape all posts
    allposts = parellelize(allsubs,allstarts,allends)
    dposts = allposts[:2]
    nposts = allposts[2:7]
    cposts = allposts[7:]

    dposts=remove_duplicates(dposts)
    nposts=remove_duplicates(nposts)

    # Pickling all posts
    pickle.dump(dposts, open(os.path.join(DATA_DIR, 'dposts.pkl'), 'wb'))
    pickle.dump(nposts, open(os.path.join(DATA_DIR, 'nposts.pkl'), 'wb'))
    pickle.dump(cposts, open(os.path.join(DATA_DIR, 'cposts.pkl'), 'wb'))

    dposts = pickle.load(open(os.path.join(DATA_DIR, 'dposts.pkl'), 'rb'))
    nposts = pickle.load(open(os.path.join(DATA_DIR, 'nposts.pkl'), 'rb'))
    cposts = pickle.load(open(os.path.join(DATA_DIR, 'cposts.pkl'), 'rb'))

    cposts1 = cposts[:int(0.5 * len(cposts))]
    cposts2 = random.sample(cposts1, int(.07 * len(cposts1)))

    subids = list(np.unique([post['id'] for post in cposts2]))
    idlen = len(subids)

    print("Extracting comments from submission id's...")
    start = time.time()
    comments_from_reddit = fetch_parallel(subids,idlen)
    comments_from_reddit = list(comments_from_reddit.queue)
    end = time.time()
    print(f"Time taken in extracting comments: {int((end - start) / 60)} minutes, {(end - start) % 60} seconds")
    print(f"Total non-depression comments: {len(comments_from_reddit)}")
    nposts = comments_from_reddit + nposts

    # Saving as dataframe
    posts = pd.DataFrame({"Text": dposts + nposts, "label": [1] * len(dposts) + [0] * len(nposts)})
    print(posts.head(5), posts.shape)
    posts.drop_duplicates().to_csv(os.path.join(DATA_DIR, 'reddit_posts.csv'), header=True, index=False,
                                   encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    print(f"Scraped data saved as csv file at: {os.path.join(DATA_DIR, 'reddit_posts.csv')}")








