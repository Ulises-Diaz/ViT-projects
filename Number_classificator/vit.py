import torch
from torch import nn
import pandas as pd
from torch import optim 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random 
import timeit
from tqdm import tqdm

random_seed = 42
batch_size = 512 
epochs = 25 #in normal transformers we just need 1 or 2 because we finetune models. In here we are training the foundation model 
learning_rate = 1e-4
num_classes = 10 #from data
patch_size = 4
img_size = 28
in_channels = 1
num_heads = 8
dropout = 0.001
hidden_dim=768
weigt_decay = 0
betas = (0.9, 0.999)
activation = "gelu"
num_encoders = 4
embed_dim = (patch_size**2) *in_channels #16
num_patches = (img_size // patch_size) ** 2 #49

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
#torch.cuda.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed)
#torch.backends.cuda.deterministic = True 
#torch.backends.cuda.benchmark = False

device = 'cuda' if torch.cuda.is_available()else "cpu"


class PatchEmbeding (nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels= in_channels,
                out_channels= embed_dim,
                kernel_size= patch_size,
                stride= patch_size
            ),
            nn.Flatten(2)
        )
        
        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad = True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad= True)
        self.dropout= nn.Dropout(p=dropout)
        
    def forward (self, x) : 
        cls_token =self.cls_token.expand(x.shape[0], -1, -1)
        x=self.patcher(x).permute(0,2,1)
        x=torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x 
        
model = PatchEmbeding(embed_dim, patch_size, num_patches, dropout, in_channels).to(device)
x= torch.randn(512, 1, 28, 28)
print(model(x).shape)


# Visual Transformer

class ViT ( nn.Module) : 
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__() 
        self.embeddings_block = PatchEmbeding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=False)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers= num_encoders)
        
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
        
    def forward (self, x): 
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :]) #only take the 0th token
        return x 

model = ViT(num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels).to(device)
x =torch.randn(512, 1, 28, 28)
print(model(x).shape)
        

#input data
train_df = pd.read_csv('/home/uli/Desktop/tec/personal/vt/train.csv')
test_df = pd.read_csv('/home/uli/Desktop/tec/personal/vt/test.csv')
submission_df = pd.read_csv('/home/uli/Desktop/tec/personal/vt/sample_submission.csv')

#train data

#print(train_df.head())
#print(test_df.head())

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=random_seed, shuffle=True)

class MNISTtrainDataset(Dataset): 
    def __init__(self, images, labels, indicies):
        self.images = images 
        self.labels = labels
        self.indices = indicies 
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)  # Aplica las transformaciones correctamente
        
        return {'Images': image, 'labels': label, "index": index}


class MNISTValDataset(Dataset) : 
    def __init__(self, images, labels, indicies):
        self.images = images 
        self.labels = labels
        self.indices = indicies 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__ (self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indices[idx]
        image = self.transform(image)          
        return {'Images': image, 'labels': label, "index": index}
    
class MNISTSubmitDataset(Dataset) : 
    def __init__(self, images, indicies):
        self.images = images 
        self.indices = indicies 
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__ (self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8)
        index = self.indices[idx]
        image = self.transform(image)  
        
        return {'Images': image, "index": index}

plt.figure ()

f, axarray = plt.subplots(1, 3) 

train_dataset = MNISTtrainDataset(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values, train_df.index.values) #Do not take labels, only pixels
print(len(train_dataset))
print(train_dataset[0])
axarray[0].imshow(train_dataset[0]["Images"].squeeze(), cmap='gray')
axarray[0].set_title("train image")
print("-"*30)

val_dataset = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values, val_df.index.values) #Do not take labels, only pixels
print(len(val_dataset))
print(val_dataset[0])
axarray[1].imshow(val_dataset[0]["Images"].squeeze(), cmap='gray')
axarray[1].set_title("Val image")
print("-"*30)

test_dataset = MNISTSubmitDataset(test_df.values.astype(np.uint8), test_df.index.values) #Do not take labels, only pixels
print(len(test_dataset))
print(test_dataset[0])
axarray[2].imshow(test_dataset[0]["Images"].squeeze(), cmap='gray')
axarray[2].set_title("Test image")
print("-"*30)

#plt.show()

train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset, 
                              batch_size=batch_size, 
                              shuffle=False)


#Training loop

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas = betas, lr = learning_rate, weight_decay= weigt_decay)

start = timeit.default_timer()
for epoch in tqdm (range(epochs), position=0, leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["Images"].float().to(device)
        label = img_label["labels"].type(torch.uint8).to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim = 1)
        
        train_labels.extend(label.cpu().detach())
        train_preds.extend(label.cpu().detach())
        
        loss = criterion(y_pred, label)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
    
    train_loss = train_running_loss/(idx+1)
    
    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0.0
    
    with torch.no_grad():
        for idx , img_label in enumerate (tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["Images"].float().to(device)
            label = img_label["labels"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim = 1)
            
            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())
            
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()
        val_loss = val_running_loss / (idx+1)
        
        
        print("-" * 30)
        
        print(f'Train loss epoch {epoch+1} : {train_loss:.4f}')
        print(f'Val loss epoch {epoch+1} : {val_loss:.4f}')
        print(f'Train accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}')
        print(f'Val accuracy EPOCH {epoch + 1}: {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}')
        print("-" *30)
        
    stop = timeit.default_timer()
    print(f' Training Time: {stop*start:.2f}s')
    
    
#Prediction Loop 

#torch.cuda.empty_cache() #to clean the memory 

labels = []
ids = []
imgs = []

model.eval()

with torch.no_grad():
    for idx, sample in enumerate(tqdm(test_dataloader, position=0 ,leave=True)):
        img = sample["Images"].to(device)
        ids.extend([int(i)+1 for i in sample["index"]])
        
        outputs = model(img)
        imgs.extend(img.detach().cpu())
        labels.extend([int(i) for i in torch.argmax(outputs, dim=1)])

plt.figure()
f ,axarray =plt.subplots(2, 3)
counter = 0
for i in range(2):
    for j in range(3): 
        axarray[i][j].imshow(imgs[counter].squeeze(), cmap ='gray')
        axarray[i][j].set_title(f'Predictet {labels[counter]}')
        counter += 1
        
plt.show()
