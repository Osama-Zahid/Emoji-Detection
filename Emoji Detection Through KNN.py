

import torch as th
from sklearn.neighbors import KNeighborsClassifier
def random_tensor(tensor):
  return tensor[th.randperm(len(tensor))]

def distance_matrix(x,y=None,p=2):
  y=x if type(y)==type(None) else y
  n=x.size(0)
  m=y.size(0)
  d=x.size(0)

  x = x.unsqueeze(1).expand(n,m,d)
  y = y.unsqueeze(0).expand(n,m,d)

  dist=th.pow(x-y,p).sum(2)

  return dist

class NN():
  def __init__(self):
      pass
  def train(self,X,Y):
    self.train_pts=X
    self.train_label=Y

  def __call__(self,x):
    return self.predict(x)
  def predict(self,x):
    if type(self.train_pts)==type(None) or type(self.train_label)==type(None):
      name=self.__class__.__name__
      raise RuntimeError()(f"{name} wasn't trained. Need to execute {name}.train() first")
      dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)
      labels = th.argmin(dist, dim=1)
      return self.train_label[labels]

class KNN():
  def __init__(self,X=None,Y=None,k=6,p=2):
    self.k=k
    super().__init__()
  def train(self,X,Y):
    super().train(X,Y)
    if type(Y) != type(None):
      self.unique_labels=self.train_label.unique()
  def predict(self,x):
    if type(self.train_pts)==type(None) or type(self.train_label)==type(None):
      name=self.__class__.__name__
    
    dist=distance_matrix(x,self.train_pts, self.p) ** (1/self.p)

    knn=dist.topk(self.k,largest=False)
    votes=self.train_label[knn.indices]

    winner=th.zeros(votes.size(0),dtype=votes.dtype,device=votes.device)
    count=th.zeros(votes.size(0),dtype=votes.dtype,device=votes.device)-1

    for lab in self.unique_labels:
      vote_counts=(votes==lab).sum(1)
      who=vote_counts>=count
      winner[who]=lab
      count[who]=vote_counts[who]
    
    return winner
BS = 20
print(X_train.shape)
for i in range(1000):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  X = th.tensor(X_train[samp])
  Y = th.tensor(Y_train[samp])

if __name__=='__main__':
  knn=KNN(X,Y)

# evaluation
Y_test_preds = torch.argmax(model(torch.tensor(X_train.reshape((-1, 100*100))).float()), dim=1).numpy()
print((Y_train == Y_test_preds).mean())
print("KNN->",knn)
#X_test = np.reshape(X_test, (X_test.shape[0], -1))
print("X_train",X_train)
#print("Y_train",Y_train)
for k in range(1000):
    nn = KNeighborsClassifier(n_neighbors=k)
    nn.fit(X_trian,)
    nn.fit(X_train, Y_train)
    #y_test_pred = nn.predict(X_test)
    print('K =', k)
print('accuracy: %f' % (np.mean(Y_test_preds == y_test)))
