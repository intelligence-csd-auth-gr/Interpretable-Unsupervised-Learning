import pickle
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

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

del texts, train_texts, test_texts, validation_texts, binarized_labels

number_of_reduced = 500

fileObj = open('predicted_instances_kpca_' + str(number_of_reduced) + '_train.obj', 'rb')
predicted_reduced_train = pickle.load(fileObj)[0]
fileObj.close()

fileObj = open('predicted_instances_kpca_' + str(number_of_reduced) + '_val.obj', 'rb')
predicted_reduced_val = pickle.load(fileObj)[0]
fileObj.close()

fileObj = open('predicted_instances_kpca_' + str(number_of_reduced) + '_.obj', 'rb')
predicted_reduced_test = pickle.load(fileObj)[0]
fileObj.close()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


input_size = (len(predicted_reduced_train[0]))
hidden_size1 = 5000
hidden_size2 = 10000
output_size = (len(train_labels[0]))

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)

train_dataset = TensorDataset(torch.tensor(predicted_reduced_train), torch.tensor(train_labels, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(predicted_reduced_val), torch.tensor(validation_labels, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(predicted_reduced_test), torch.tensor(test_labels, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters())

epochs = 100
patience = 20
best_val_loss = float('inf')
counter = 0

for epoch in range(epochs):  # you can adjust the number of epochs
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc='Training'):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='Validation'):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model, 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping')
            break


def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Testing'):
            x = x.to(device)
            y_pred = model(x)
            y_pred = y_pred.cpu()
            predictions.append(y_pred.detach().numpy())
    return predictions


predictions = predict(model, test_loader)
fileObj = open('predictions_of_exml_' + str(number_of_reduced) + '.obj', 'wb')
pickle.dump([predictions, test_labels], fileObj)
fileObj.close()
