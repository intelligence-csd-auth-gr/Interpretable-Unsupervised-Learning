import numpy as np
from bert_multi_label import BertForMultilabelSequenceClassification
from my_dataset import myDataset
from p_at_k import precision_at_k_score
from sklearn.metrics import f1_score
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

train_dataset = myDataset(train_texts, train_labels, tokenizer)
validation_dataset = myDataset(validation_texts, validation_labels, tokenizer)
test_dataset = myDataset(test_texts, test_labels, tokenizer)

model = BertForMultilabelSequenceClassification.from_pretrained("bert_mesh_all/", num_labels=number_of_labels,
                                                                output_attentions=False, output_hidden_states=False)
model.to('cuda')
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset)
trainer.train()

model.save_pretrained('bert_mesh_all/')

outputs = trainer.predict(test_dataset)
predictions = outputs.predictions

pred_labels = list()
for prediction in predictions:
    pred_labels.append([1 if i >= 0 else 0 for i in prediction])

del prediction, trainer, model, train_dataset, train_texts, train_labels, test_dataset, test_texts
del validation_dataset, validation_texts, validation_labels, outputs

pred_labels = np.array(pred_labels)
print(f1_score(np.array(test_labels), np.array(pred_labels), average='macro'))

del pred_labels

predictions_temp = list()
for prediction in predictions:
    predictions_temp.append(list(prediction))
predictions = predictions_temp.copy()

del predictions_temp

print(precision_at_k_score(test_labels.A, predictions))
