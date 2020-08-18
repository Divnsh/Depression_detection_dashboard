from transformers import (TFRobertaForSequenceClassification,
                          RobertaTokenizer)
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,classification_report
#import tensorflow_datasets as tfds
import logging
logging.basicConfig(level=logging.ERROR)
from transformers import RobertaConfig

output_dir = './roBERTamodels/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
colastext = 'cleaned_text'
max_length=256

class Model:
    def __init__(self):
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.config=RobertaConfig.from_json_file(output_dir + 'config.json')
        self.model=TFRobertaForSequenceClassification.from_pretrained(output_dir+'tf_model.h5',config=self.config)
        # self.model._make_predict_function()
        # self.session = tf.keras.backend.get_session()
        # self.graph = tf.get_default_graph()

    def get_prediction(self,data,batch_size=1):
        assert os.path.exists(output_dir+'tf_model.h5') and os.path.exists(output_dir+'config.json'),"Trained model not found. Please train a model first!"
        try:
            id_list = []
            attention_list = []
            for post in data[colastext]:
                tokens = self.roberta_tokenizer.encode_plus(post,
                                                       add_special_tokens=True,  # add [CLS], [SEP]
                                                       max_length=max_length,  # max length of the text that can go to RoBERTa
                                                       pad_to_max_length=True,  # add [PAD] tokens at the end of sentence,
                                                       truncation=True,
                                                       return_attention_mask=True,
                                                       # add attention mask to not focus on pad tokens
                                                       )
                id_list.append(tokens['input_ids'])
                attention_list.append(tokens['attention_mask'])

            def map_to_dict(input_ids, attention_masks):
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                }

            test_encoded = tf.data.Dataset.from_tensor_slices((id_list, attention_list)).map(map_to_dict)
            test_encoded = test_encoded.batch(batch_size)
            test_preds = self.model.predict(test_encoded)
            test_preds = tf.nn.softmax(test_preds)
            test_preds1 = tf.reshape(test_preds, [test_preds.shape[1], 2])
            preds=test_preds1[:, 1].numpy()
            return preds
        except Exception as e:
            print(e)
            return ["GPU unavailable"]

if __name__=='__main__':
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    output_dir = './roBERTamodels/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    colastext = 'cleaned_text'
    # colastext='stemmed_text'
    # colastext='lemmatized_text'
    # colastext='treated_stemmed_text'
    # colastext='treated_lemmatized_text'

    DATA_DIR='./Data'
    fulldata=pd.read_csv(os.path.join(DATA_DIR,"cleaned_reddit.csv"))
    fulldata=fulldata.drop_duplicates().dropna()

    train, test, train_labels, test_labels = train_test_split(fulldata[[colastext]], fulldata['label'],
                                            random_state=2020, test_size=0.30, stratify=fulldata['label'])

    val, test, val_labels, test_labels = train_test_split(test, test_labels,
                                            random_state=2020, test_size=0.50)
    del fulldata

    # Using TPU architecture
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    AUTO = tf.data.experimental.AUTOTUNE
    print("Number of accelerators: ", strategy.num_replicas_in_sync)

    max_length = 256
    batch_size = 32*strategy.num_replicas_in_sync
    #batch_size = 32
    n_training_examples=len(train)
    STEPS_PER_EPOCH=n_training_examples//batch_size

    def convert_example_to_feature(review):
        # combine step for tokenization, WordPiece vector mapping and will
        # add also special tokens and truncate reviews longer than our max length
        bert_input = roberta_tokenizer.encode_plus(review,
                                     add_special_tokens=True,  # add [CLS], [SEP]
                                     max_length=max_length,  # max length of the text that can go to RoBERTa
                                     pad_to_max_length=True,  # add [PAD] tokens at the end of sentence,
                                     truncation=True,
                                     return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                     )
        return bert_input

    # map to the expected input to TFRobertaForSequenceClassification, see here
    def map_example_to_dict(input_ids, attention_masks, label):
        return {
          "input_ids": input_ids,
          "attention_mask": attention_masks,
               }, label

    def encoding_collate(input_ids_list,attention_mask_list,label_list):
        return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                                   attention_mask_list,
                                                   label_list)).map(map_example_to_dict)

    def get_encoded(dat,labels):
        train_input_ids_list=[]
        train_attention_mask_list=[]
        train_label_list=[]
        for train_split,train_labels_split in zip(np.array_split(dat,5),np.array_split(labels,5)):
            training_sentences_modified = train_split[colastext].apply(convert_example_to_feature)
            train_input_ids_list.extend(list(training_sentences_modified.apply(lambda x:x['input_ids'])))
            train_attention_mask_list.extend(list(training_sentences_modified.apply(lambda x:x['attention_mask'])))
            lab_list=[[label] for label in train_labels_split]
            train_label_list.extend(lab_list)
        ds_encoded=encoding_collate(train_input_ids_list,train_attention_mask_list,train_label_list)
        return ds_encoded

    ds_train_encoded=get_encoded(train,train_labels).repeat().shuffle(10000).batch(batch_size).prefetch(AUTO)

    ds_val_encoded=get_encoded(val,val_labels).batch(batch_size).cache().prefetch(AUTO)

    ds_test_encoded=get_encoded(test,test_labels).batch(batch_size)

    import gc
    del train,test,val
    gc.collect()

    learning_rate = 6e-5
    number_of_epochs = 3

    # model initialization

    # we do not have one-hot vectors, we can use sparse categorical cross entropy and accuracy

    def init_model():
        with strategy.scope():
            model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        return model

    model=init_model()

    history=model.fit(ds_train_encoded, epochs=number_of_epochs,steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=ds_val_encoded)

    # Evaluation on test set
    model.evaluate(ds_test_encoded)

    # Saving trained model

    # Can be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model
    print("Saving model to %s" % output_dir)
    model_to_save.save_pretrained('/kaggle/working/')

    # Further evaluation
    config = RobertaConfig.from_json_file(output_dir+'config.json')
    model = TFRobertaForSequenceClassification.from_pretrained(output_dir+'tf_model.h5',config=config)
    id_list=[]
    attention_list=[]
    for post in test[colastext]:
        tokens=roberta_tokenizer.encode_plus(post,
                                         add_special_tokens=True,  # add [CLS], [SEP]
                                         max_length=max_length,  # max length of the text that can go to RoBERTa
                                         pad_to_max_length=True,  # add [PAD] tokens at the end of sentence,
                                         truncation=True,
                                         return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                         )
        id_list.append(tokens['input_ids'])
        attention_list.append(tokens['attention_mask'])

    def map_to_dict(input_ids, attention_masks):
        return {
          "input_ids": input_ids,
          "attention_mask": attention_masks,
               }
    test_encoded=tf.data.Dataset.from_tensor_slices((id_list,attention_list)).map(map_to_dict)
    test_encoded=test_encoded.batch(32)
    test_preds=model.predict(test_encoded)

    test_preds = tf.nn.softmax(test_preds)
    test_preds1=tf.reshape(test_preds, [test_preds.shape[1], 2])
    test_preds_argmax = tf.math.argmax(test_preds1, axis=1)
    roc_auc = roc_auc_score(test_labels,test_preds_argmax)
    print('\n test roc_auc score is :', roc_auc)
    print("\n test classification report: ",classification_report(test_labels,test_preds_argmax))

