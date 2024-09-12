import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from data_class import *
import random

device = 'cpu'

class test_model(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(test_model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        test_layers = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        out = torch.zeros((self.out_dim, x.shape[-1]))
        for i in range(out.shape[-1]):
            out[:, i] = self.test_layers(x[:, i])


#chord models
class test_LSTM(nn.Module):
    #input is a list of chord indices
    def __init__(self, in_dim, num_layers, embedding, out_dim = num_chords):
        super(test_LSTM, self).__init__()
        assert in_dim == embedding.embedding_dim, 'embedding_dim != in_dim!'

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embedding = embedding
        self.lstm = nn.LSTM(in_dim, out_dim, 1, batch_first=True)

    def forward(self, x):
        out = self.embedding(x)
        print(out.shape)
        out = self.lstm(out)
        return out



class test_transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, max_len=5000, dropout=0.1):
        super(test_transformer, self).__init__()
        #self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = nn.Embedding(max_len, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim, dropout=dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1]).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = self.embedding(x.float()) + self.pos_encoder(positions)

        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x


class test_embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(test_embedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        print(x[:10])
        out = self.embedding(x)
        print('embedded', out.shape)
        return out


class VarLenChordDataset(Dataset):

    def __init__(self, data_list):
        super(VarLenChordDataset, self).__init__()

        self.chords_list = data_list

    def __len__(self):
        return len(self.chords_list)

    def __getitem__(self, idx):
        frame_length = random.randint(1,len(self.chords_list[idx])-1)

        item_x = self.chords_list[idx][:frame_length]
        item_y = self.chords_list[idx][1:frame_length+1]

        return torch.tensor(item_x), torch.tensor(item_y)



def train(model, criterion, optimizer, epochs, dl):
    model.train()

    for epoch in range(epochs):
        batch_losses = []
        batch_accuracies = []

        for x,y in dl:

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            out, _ = model.forward(x)
            print('x',x.shape,'out',out.shape)

            loss = criterion(torch.permute(out, (0,2,1)), y)
            loss.backward()

            if isinstance(model, nn.TransformerEncoder):
                nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            batch_losses.append(loss.item())

        print('Local_Loss: ', round(sum(batch_losses)/len(batch_losses),4))


len_train_set = 100
len_test_set = 10

train_set, test_set = get_chord_train_and_test_set(len_train_set, len_test_set)

ds = VarLenChordDataset(train_set)
dl = DataLoader(ds)
epochs = 1
model = test_LSTM(chord_embedding_dim, num_layers=2, embedding=test_embedding(num_chords, chord_embedding_dim))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

train(model, criterion, optimizer, epochs, dl)

for i, song in enumerate(ds):
    print(len(song[1]))