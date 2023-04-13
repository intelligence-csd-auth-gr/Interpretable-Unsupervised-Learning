import pickle

import numpy as np
from bert_multi_target import BertForMultiTargetSequenceRegression
from my_dataset import myDataset
from scipy.sparse import csr_matrix
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast
from transformers import TrainingArguments, Trainer

files = open('mesh_2020_test.txt')
texts = []
labels = []
counter = 0
for line in files:
    texts.append(line[2:-2].split('abstracttt: ')[1].split('labels: ')[0])
    labels.append(line[2:-2].split('abstracttt: ')[1].split('labels: ')[1][1:].split('#'))
files.close()

mlb = MultiLabelBinarizer()
mlb.fit(labels)

binarized_labels = mlb.transform(labels)

number_of_labels = len(mlb.classes_)

del mlb, labels, files

train_texts, test_texts, train_labels, test_labels, = train_test_split(texts, binarized_labels, test_size=.2,
                                                                       random_state=42)

size = (0.1 * len(binarized_labels)) / len(train_labels)
train_texts, validation_texts, train_labels, validation_labels = train_test_split(list(train_texts), train_labels,
                                                                                  test_size=size, random_state=42)

del texts, binarized_labels

training_args = TrainingArguments(
    evaluation_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
    log_level='warning', output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=4, per_device_eval_batch_size=4,
    warmup_steps=200, weight_decay=0.001, logging_dir='./logs',
    save_total_limit=1, load_best_model_at_end=True,
)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

number_of_reduced = 500  # try with 5000 as well

kpca = KernelPCA(kernel='rbf', n_components=number_of_reduced, random_state=42)
kpca.fit(csr_matrix(train_labels))
fileObj = open('kpca_' + str(number_of_reduced) + '.obj', 'wb')
pickle.dump(kpca, fileObj)
fileObj.close()

train_labels_reduced1 = kpca.transform(train_labels[:10000])
train_labels_reduced2 = kpca.transform(train_labels[10000:])
train_labels_reduced = np.concatenate([train_labels_reduced1, train_labels_reduced2])
del train_labels_reduced1, train_labels_reduced2
validation_labels_reduced = kpca.transform(validation_labels)
test_labels_reduced = kpca.transform(test_labels)
print(test_labels_reduced.min(), test_labels_reduced.max())

mms = StandardScaler()
mms.fit(train_labels_reduced)

train_labels_reduced = mms.transform(train_labels_reduced)
validation_labels_reduced = mms.transform(validation_labels_reduced)
test_labels_reduced = mms.transform(test_labels_reduced)

train_dataset = myDataset(train_texts, train_labels_reduced, tokenizer)
validation_dataset = myDataset(validation_texts, validation_labels_reduced, tokenizer)
test_dataset = myDataset(test_texts, test_labels_reduced, tokenizer)

model = BertForMultiTargetSequenceRegression.from_pretrained("bert-base-uncased", num_labels=number_of_reduced,
                                                             output_attentions=False, output_hidden_states=False)
model.to('cuda')
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset)
model.save_pretrained('bert_mesh_kpca_reduced_500/')

outputs = trainer.predict(train_dataset)
predictions = outputs.predictions

fileObj = open('predicted_instances_kpca_500_train.obj', 'wb')
pickle.dump([predictions, mms], fileObj)
fileObj.close()

outputs = trainer.predict(validation_dataset)
predictions = outputs.predictions

fileObj = open('predicted_instances_kpca_500_val.obj', 'wb')
pickle.dump([predictions, mms], fileObj)
fileObj.close()

outputs = trainer.predict(test_dataset)
predictions = outputs.predictions

fileObj = open('predicted_instances_kpca_500.obj', 'wb')
pickle.dump([predictions, mms], fileObj)
fileObj.close()
