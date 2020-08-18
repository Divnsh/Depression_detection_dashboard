import torch
#import tensorflow as tf
import pandas as pd
import os
from transformers import BertTokenizer
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

DATA_DIR='./Data'
fulldata=pd.read_csv(os.path.join(DATA_DIR,"cleaned_reddit.csv"))
fulldata=fulldata.drop_duplicates().dropna()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Split into train and test
train, test, train_labels, test_labels = train_test_split(fulldata, fulldata['label'],
                                        random_state=2020, test_size=0.15, stratify=fulldata['label'])
del fulldata

colastext='cleaned_text'
# colastext='stemmed_text'
# colastext='lemmatized_text'
# colastext='treated_stemmed_text'
# colastext='treated_lemmatized_text'

def text_processing(data):
    input_ids = []
    lengths = []
    print('Tokenizing...')
    MAX_LEN = 512

    for post in data[colastext]:
        if ((len(input_ids) % 20000) == 0):
            print('  Read {:,} posts.'.format(len(input_ids)))

        ids = tokenizer.encode(
            post,  # Sentences to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            #max_length=MAX_LEN
        )

        input_ids.append(ids)
        lengths.append(len(ids))

    print('DONE.')
    print('{:>10,} posts'.format(len(input_ids)))

    labels = data.label.to_numpy()

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (10,5)

    # Truncate any comment lengths greater than 512.
    lengths = [min(l, 512) for l in lengths]

    # Plot the distribution of comment lengths.
    sns.distplot(lengths, kde=False, rug=False)
    plt.title('Post Lengths')
    plt.xlabel('Post Length')
    plt.ylabel('# of Posts')
    plt.show()

    prcnt =lengths.count(512)/ len(lengths)
    print('{:.1%} posts in the data are longer than 512 tokens.'.format(prcnt))

    # Padding
    print('\nTruncating all sentences to %d values...' % MAX_LEN)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    print('\nDone.')

    # Create attention masks
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return input_ids,labels,attention_masks

input_ids,labels,attention_masks=text_processing(train)

# Splitting into train, val, test sets 72-13-15
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                        random_state=2020, test_size=0.15, stratify=labels)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                        random_state=2020, test_size=0.15, stratify=labels)


# Convert all inputs and labels into torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 32
# DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # 12-layer BERT model, with an uncased vocab.
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.mean(pred_flat == labels_flat)

# Set seed for reproducibility.
seed_val = 2020
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = [] # average loss per epoch

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader) # avg loss
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation...")
    t0 = time.time()

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

plt.plot(loss_values, 'b-o')
plt.title("Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

test_inputs,test_labels,test_masks=text_processing(test)

batch_size = 32
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

print('Predicting for {:,} test posts...'.format(len(test_inputs)))

model.eval()
predictions, true_labels = [], []
t0 = time.time()

for (step, batch) in enumerate(test_dataloader):

    batch = tuple(t.to(device) for t in batch)
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')

predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

p1 = predictions[:,1]
auc = roc_auc_score(true_labels, p1)
print('Test ROC AUC: %.3f' %auc)

# Saving trained model
output_dir = './BERTmodels/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving model to %s" % output_dir)

# Can be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
#torch.save(args, os.path.join(output_dir, 'training_args.bin'))



