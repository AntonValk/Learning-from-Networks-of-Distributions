# testMLP
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
color = sns.color_palette()

# %matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'


train_df = pd.read_csv('neighbourhood_data.csv')
districts = pd.read_csv('districts_data_cleaned.csv')
names = list(districts['name'].unique())
train_df = train_df[train_df['neighbourhood'].isin(names)]
counts = train_df['neighbourhood'].value_counts()
count_list = counts[counts > 50].index.tolist()
train_df = train_df[train_df['neighbourhood'].isin(count_list)]


train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
train_df = train_df.astype({'bathrooms': 'int64'})
train_df = train_df.astype({'bedrooms': 'int64'})
train_df = train_df.astype({'interest_level': 'category'})
train_df = train_df.astype({'num_photos': 'int64'})
train_df = train_df.astype({'num_features': 'int64'})
train_df = train_df.astype({'num_description_words': 'int64'})
train_df = train_df.astype({'created_month': 'category'})
train_df = train_df.astype({'created_day': 'category'})
train_df = train_df.astype({'neighbourhood': 'str'})
train_df = train_df.astype({'price': 'float64'})
non_standardized = train_df.copy()
# train_df['price'] = train_df['price'].clip(upper = np.percentile(train_df['price'].values, 99))
train_df['price'] = (train_df['price'] - train_df['price'].mean()) / train_df['price'].std()



def train_Set(n = 25):
    train_samples = train_df.groupby('neighbourhood').apply(pd.DataFrame.sample, n, replace = False)
    train_samples = train_samples.astype({'price': 'float64'})
    features = []
    for i, hood in enumerate(train_samples['neighbourhood'].unique()):
        get_boro = train_samples[train_samples['neighbourhood'] == hood]
        sample = get_boro.to_numpy()
        features.append(sample)
    return np.array(features)


def sampleSet(n = 25, size = 1000):
    train_samples = train_df.groupby('neighbourhood').apply(pd.DataFrame.sample, n, replace = True)
    train_samples = train_samples.astype({'price': 'float64'})
    features = []
    labels = []
    for i, hood in enumerate(train_samples['neighbourhood'].unique()):
        samples = []
        label = []
        get_boro = train_samples[train_samples['neighbourhood'] == hood]
        get_boro = get_boro['price']
        for j in range(size):
            sample = get_boro.sample(n, replace = True).to_numpy()
            samples.append(sample)
            label.append(np.mean(sample))
        features.append(samples)
        labels.append(label)
    return np.array(features), np.array(labels)


X, y = sampleSet()
X_train = X
y_train = y



features = np.reshape(X_sets,(X_sets.shape[0], -1))
features = np.array(features, dtype='float')
features = torch.FloatTensor(features)
labels = torch.FloatTensor(mean)
adj = torch.FloatTensor(adj)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

class MLP(Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = x
        return output



EPOCHS = 1500

# Model and optimizer
model2 = MLP()
# model = GCN(nfeat=features.shape[1],nclass=1,dropout=0.2)
optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=0.001)
mse = nn.MSELoss()
# mse = nn.L1Loss()




def train(epoch):
    t = time.time()
    model2.train()
    optimizer.zero_grad()
    output = model2(features)
    loss_train = mse(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = mse(output[idx_val], labels[idx_val])
    if((epoch + 1) % 500 == 0):
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def test():
    model2.eval()
    output = model2(features)
    loss_test = mse(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
for epoch in range(EPOCHS):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

myCsvRow = non_standardized['price'].std() * np.sqrt(mse(output[idx_test], labels[idx_test]).detach().numpy())


with open('mlp_tests.csv','a') as fd:
    fd.write(myCsvRow)