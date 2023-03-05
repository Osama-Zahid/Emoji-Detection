
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False)
class BobNet(torch.nn.Module):
  def __init__(self):
    super(BobNet, self).__init__()
    self.l1 = nn.Linear(100*100, 500, bias=True)
    self.l2 = nn.Linear(500, 6, bias=True)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x

# training
model = BobNet()

loss_function = nn.NLLLoss(reduction='none')
optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
BS = 20
losses, accuracies = [], []
for i in range(1000):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  X = torch.tensor(X_train[samp].reshape((-1, 100*100))).float()
  Y = torch.tensor(Y_train[samp])
  model.zero_grad()
  out = model(X)
  cat = torch.argmax(out, dim=1)
  accuracy = (cat == Y).float().mean()
  loss = loss_function(out, Y)
  loss = loss.mean()
  loss.backward()
  optim.step()
  loss, accuracy = loss.item(), accuracy.item()
  losses.append(loss)
  accuracies.append(accuracy)
  if i % 100 == 0:
    print("loss %.2f accuracy %.2f" % (loss, accuracy))
plt.ylim(-0.1, 1.1)
plot(losses)
plot(sorted(accuracies))

# evaluation
Y_test_preds = torch.argmax(model(torch.tensor(X_train.reshape((-1, 100*100))).float()), dim=1).numpy()
(Y_train == Y_test_preds).mean()

#max accuracy attained
plot(sorted(accuracies))