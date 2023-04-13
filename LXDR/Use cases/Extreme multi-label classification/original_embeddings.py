import pickle

from my_dataset import myDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
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

model = BertForMultilabelSequenceClassification.from_pretrained("bert_mesh_all/", num_labels=number_of_labels,
                                                                output_attentions=False, output_hidden_states=True)
model.to('cuda')
trainer = Trainer(model=model)

embeddings_train = []
for i in range(len(train_texts)):
    train_dataset = myDataset([train_texts[i]], [train_labels[i]], tokenizer)
    outputs = trainer.predict(train_dataset)
    embeddings_train.append(outputs[0][1][-1][0][0])

fileObj = open('embeddings_train.obj', 'wb')
pickle.dump(embeddings_train, fileObj)
fileObj.close()

del embeddings_train, train_texts

embeddings_test = []
for i in range(len(test_texts)):
    test_dataset = myDataset([test_texts[i]], [test_labels[i]], tokenizer)
    outputs = trainer.predict(test_dataset)
    embeddings_test.append(outputs[0][1][-1][0][0])

fileObj = open('embeddings_test.obj', 'wb')
pickle.dump(embeddings_test, fileObj)
fileObj.close()
