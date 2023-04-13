import pickle

from bert_multi_target import BertForMultiTargetSequenceRegression
from my_dataset import myDataset
from sklearn.decomposition import PCA
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

number_of_reduced = 5000
pca = PCA(n_components=number_of_reduced, random_state=42)
pca.fit(train_labels)
fileObj = open('pca_' + str(number_of_reduced) + '.obj', 'wb')
pickle.dump(pca, fileObj)
fileObj.close()

train_labels_reduced = pca.transform(train_labels)
validation_labels_reduced = pca.transform(validation_labels)
test_labels_reduced = pca.transform(test_labels)

print(sum(pca.explained_variance_ratio_))

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
trainer.train()
model.save_pretrained('bert_mesh_5000_pca_reduced_zscale/')

predictions = trainer.predict(test_dataset)
predictions = predictions.predictions
predictions = mms.inverse_transform(predictions)
predictions = pca.inverse_transform(predictions)

fileObj = open('pca_5000_predictions.obj', 'wb')
pickle.dump([predictions, test_labels], fileObj)
fileObj.close()
