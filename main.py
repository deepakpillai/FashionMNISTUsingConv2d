import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.nn.modules.pooling import MaxPool2d

traindata = datasets.FashionMNIST('data', train=True, transform=transforms.ToTensor(), download=True)
testdata = datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor(), download=True)
traindata_classes = traindata.classes

train_batch = DataLoader(traindata, batch_size=32, shuffle=True)
test_batch = DataLoader(testdata, batch_size=32, shuffle=True)


class CNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_one_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv_two_layer = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.linear_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=160, out_features=10)
    )

  def forward(self, x):
    y = self.conv_one_layer(x)
    y = self.conv_two_layer(y)
    y = self.linear_layer(y)
    return y


#Sample model object
model = CNNModel()
print(model)

#testing model output using random numbers
rand_image = torch.randn(size=(1, 28,28))
pred = model(rand_image.unsqueeze(0))
print(pred)

#testing model output using random numbers in a 32 batch size
rand_image = torch.randn(size=(32, 1, 28,28))
pred = model(rand_image)
print(pred)

#Training model
number_of_epoch = 6
model = CNNModel()
loss_fn = nn.CrossEntropyLoss()
optim_fn = torch.optim.SGD(params=model.parameters(), lr=0.01)
for epoch in tqdm(range(0, number_of_epoch)):
  for batch, (feature, label) in enumerate(train_batch):
    model.train()
    pred = model(feature)
    loss = loss_fn(pred, label)
    optim_fn.zero_grad()
    loss.backward()
    optim_fn.step()

    if batch % 400 == 0:
      pred_numpy = pred.argmax(1).detach().numpy()
      label_numpy = label.detach().numpy()
      # print(pred_numpy)
      # print(label_numpy)
      print(f"loss {loss} accuracy {accuracy_score(pred_numpy, label_numpy)}")

#Testing model with a random value from test data
random_number = random.randint(0, len(testdata))
feature, label = testdata[random_number]
print(label)
logits = model(feature.unsqueeze(0))
preds = torch.argmax(logits)
print(preds.item())

def compare_predictions(input_list):
  pred = []
  with torch.inference_mode():
    for sample in input_list:
      model.eval()
      logits = model(sample)
      softmax = nn.Softmax(dim=1)
      pred_probs = softmax(logits)
      preds = pred_probs.argmax()
      pred.append(preds.item())
  return pred


#take random items from testdata to check the model
test_features = []
test_labels = []
for feaure, label in random.sample(list(testdata), k=9):
  test_features.append(feaure.unsqueeze(0))
  test_labels.append(label)

preds = compare_predictions(test_features)
print(f"Actual labels: {test_labels}")
print(f"Predicted labels: {preds}")


plt.figure(figsize=(12,12))
for index, features in enumerate(test_features):
  plt.subplot(3,3, index+1)
  plt.imshow(features.squeeze(), cmap='gray')
  color = 'g' #green
  if test_labels[index] != preds[index]:
    color = 'r' #green
  title = f"True: {traindata_classes[test_labels[index]]} | Pred: {traindata_classes[preds[index]]}"
  plt.title(title, c=color)
plt.show()