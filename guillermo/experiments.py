import numpy as np 
import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader 
import torch.optim as optim 
from sklearn import preprocessing
from sklearn.svm import SVC
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import math

###################################################
################ DATASET CLASS ####################
###################################################
class my_dataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

###################################################
############ LOCALLY CONNECTED LAYER ##############
###################################################

class LocallyConnected1D(nn.Module):
    def __init__(self, field_size, outputs_per_field, stride):
        super().__init__()

        self.field_size = field_size
        self.outputs_per_field = outputs_per_field
        self.stride = stride

        #initialized later
        self.weights = None
        self.bias = None
        self.n_fields = None
        self.input_size = None

    def _initialize_parameters(self, input_size, device):
        self.input_size = input_size

        n_fields = math.ceil((input_size - self.field_size) / self.stride) + 1
        self.n_fields = n_fields

        self.weights = nn.Parameter((1 / np.sqrt(self.field_size))*torch.randn(n_fields, self.field_size, self.outputs_per_field, device=device))

        self.bias = nn.Parameter(torch.zeros((n_fields, self.outputs_per_field), device=device))

    def forward(self, x):
        # x shape: (batch, length)

        if self.weights is None:
            self._initialize_parameters(x.size(1), x.device)

        L_padded = (self.n_fields - 1) * self.stride + self.field_size
        pad = L_padded - self.input_size

        y = F.pad(x, (0, pad))
        patches = y.unfold(1, self.field_size, self.stride)

        out = (
            torch.einsum("bni,nio->bno", patches, self.weights)
            + self.bias
        )

        return out.reshape(x.size(0), -1)


###################################################
################ TRAINING LOOP ####################
###################################################

def training_model(model, X_train, X_test, device='cpu', n_epochs=50):
    
    #input dimensionality 
    d = X_train.shape[1]
    
    #creating datasets
    train_dataset = my_dataset(X_train, X_train)
    test_dataset = my_dataset(X_test, X_test)
    
    #creating dataloader 
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    
    #optimization details
    criterion = nn.MSELoss()
    model.eval()
    model(train_dataset[0][0].unsqueeze(0)) 
    optimizer = optim.Adam(model.parameters())

    #training loop
    for epoch in range(n_epochs):
        
        model.train()
        for inputs, targets in train_dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            #forward pass
            outputs = model(inputs)
            
            #loss calculation
            loss = criterion(outputs[:, :d], targets)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
    #model evaluation 
    model.eval()
    with torch.no_grad():
        Z_test = model.encoder(test_dataset.X)
        loss_test = criterion(model(test_dataset.X)[:, :d], test_dataset.X)

    with torch.no_grad():
        Z_train = model.encoder(train_dataset.X)

    return Z_train, Z_test, loss_test 


###################################################
################ CLASSIFICATION ###################
###################################################

def classification_pipeline(Z_train, Z_test, y_train, y_test):
    
    #feature scaling 
    aux_train = (Z_train - torch.mean(Z_train, dim = 0))/torch.std(Z_train, dim = 0)
    aux_test = (Z_test - torch.mean(Z_train, dim = 0))/torch.std(Z_train, dim = 0)

    #svm classifier 
    classifier = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    classifier.fit(aux_train.numpy(), y_train.numpy())
    score = balanced_accuracy_score(y_test.numpy(), classifier.predict(aux_test.numpy()))

    return score

###################################################
#################### MODELS #######################
###################################################

class LC_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            LocallyConnected1D(field_size=20, outputs_per_field=10, stride=20),
            nn.ReLU(),
            LocallyConnected1D(field_size=20, outputs_per_field=10, stride=20), 
            nn.ReLU(), 
            LocallyConnected1D(field_size=500, outputs_per_field=50, stride=500),
            nn.ReLU(),
            LocallyConnected1D(field_size=50, outputs_per_field=10, stride=50),
        )
        self.decoder = nn.Sequential(
            LocallyConnected1D(field_size=10, outputs_per_field= 50, stride=10),
            nn.ReLU(),
            LocallyConnected1D(field_size=50, outputs_per_field=500, stride=50),
            nn.ReLU(),
            LocallyConnected1D(field_size=10, outputs_per_field=20, stride=10),
            nn.ReLU(), 
            LocallyConnected1D(field_size=10, outputs_per_field=20, stride=10),
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


class dense_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=2000, out_features=28),
            nn.ReLU(),
            nn.Linear(in_features=28, out_features=10)  
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=10, out_features=28), 
            nn.ReLU(), 
            nn.Linear(in_features=28, out_features=2000)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    

###################################################
################# LOADING DATA ####################
###################################################

#data
X_train = np.loadtxt('data\X_train.txt').astype(np.float32)
y_train = np.loadtxt('data\y_train.txt').astype(np.int64)-1
X_test = np.loadtxt('data\X_test.txt').astype(np.float32)
y_test = np.loadtxt('data\y_test.txt').astype(np.int64)-1

#feature scaling 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#converting to tensors
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

#feature ordering 
mi_ordering = np.loadtxt('feature_ordering\mi_ordering.txt')-1
correlation_ordering = np.loadtxt('feature_ordering\correlation_ordering.txt')-1

###################################################
################# TRAINING RUNS ###################
###################################################

n_runs = 20

#losses 
test_loss_mi_ordering = []
test_loss_correlation_ordering = []
test_loss_random_ordering = []
test_loss_dense = []

#accuracies 
accuracies_mi_ordering = []
accuracies_correlation_ordering = []
accuracies_random_ordering = []
accuracies_dense = []

for i in range(n_runs):

    #mi_ordering 
    model = LC_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    Z_train, Z_test, loss_test = training_model(model, X_train[:, mi_ordering], X_test[:, mi_ordering])
    test_loss_mi_ordering.append(loss_test)
    accuracies_mi_ordering.append(classification_pipeline(Z_train, Z_test, y_train, y_test))

    #correlation ordering
    model = LC_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    Z_train, Z_test, loss_test = training_model(model, X_train[:, correlation_ordering], X_test[:, correlation_ordering])
    test_loss_correlation_ordering.append(loss_test)
    accuracies_correlation_ordering.append(classification_pipeline(Z_train, Z_test, y_train, y_test))

    #random ordering
    model = LC_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    random_ordering = np.argsort(np.random.rand(X_train.shape[1]))
    Z_train, Z_test, loss_test = training_model(model, X_train[:, random_ordering], X_test[:, random_ordering])
    test_loss_random_ordering.append(loss_test)
    accuracies_random_ordering.append(classification_pipeline(Z_train, Z_test, y_train, y_test))

    #dense model 
    model = dense_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    Z_train, Z_test, loss_test = training_model(model, X_train, X_test)
    test_loss_dense.append(loss_test)
    accuracies_dense.append(classification_pipeline(Z_train, Z_test, y_train, y_test))


#saving results 
np.savetxt('results/test_loss_mi_ordering.txt', test_loss_mi_ordering)
np.savetxt('results/test_loss_correlation_ordering.txt', test_loss_correlation_ordering)
np.savetxt('results/test_loss_random_ordering.txt', test_loss_random_ordering)
np.savetxt('results/test_loss_dense.txt', test_loss_dense)
np.savetxt('results/accuracies_mi_ordering.txt',accuracies_mi_ordering)
np.savetxt('results/accuracies_correlation_ordering.txt', accuracies_correlation_ordering)
np.savetxt('results/accuracies_random_ordering.txt', accuracies_random_ordering)
np.savetxt('results/accuracies_dense.txt', accuracies_dense)