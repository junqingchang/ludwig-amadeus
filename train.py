import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import pretty_midi
import csv
import numpy as np
import random
import json
from models import *
import sys


maestro_dir = 'maestro-v2.0.0/'  # maestro-v2.0.0 folder with midi
output_model_dir = 'checkpoint/'
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)
features_dir = 'data/'
if not os.path.exists(features_dir):
    os.mkdir(features_dir)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Maestro(Dataset):
    '''
    Maestro Dataset
    Inputs:
        path (str): path to maestro folder
        processed_data (str): path to processed files
        train_val_test (str): one of 'train', 'validation', or 'test'. Indicates which dataset to use. Defaults: 'train'
        truncate (int or None): How much data to use, None if using all. Defaults: None
        max_seq (int): Sequence length for input. Defaults: 200
        raw (boolean): Decides if to read raw data. Defaults: False
    '''

    def __init__(self, path, processed_data, train_val_test='train', truncate=None, max_seq=200, raw=False):
        self.path = path
        self.processed_data = processed_data
        self.train_val_test = train_val_test
        # Ensures split type is correct
        assert self.train_val_test == 'train' or self.train_val_test == 'validation' or self.train_val_test == 'test'
        self.info = []  # holds all the data, data in the form of dictionary with keys: 'composer', 'title', 'year', 'midi_name', 'audio_name', 'duration'
        self.max_seq = max_seq
        with open(self.path + 'maestro-v2.0.0.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)  # csv reader
            # Omit header line
            first_line = True
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                composer, title, split, year, midi_name, audio_name, duration = line
                if split == self.train_val_test:
                    # Add data into dataset
                    self.info.append({'composer': composer, 'title': title, 'year': year,
                                      'midi_name': midi_name, 'audio_name': audio_name, 'duration': duration})

        if raw and train_val_test == 'train':
            self.ntoi = {0: 0}  # Dictionary to map the notes
            self.iton = {0: 0}

            self.idx_counter = 1  # Counter for notes mapping

            # Notes finding from midi file
            for i, music in enumerate(self.info):
                if truncate:
                    if i >= truncate:
                        break
                midi_file = pretty_midi.PrettyMIDI(
                    self.path + music['midi_name'])
                piano_midi = midi_file.instruments[0]
                piano_array = piano_midi.get_piano_roll(fs=30)
                times = np.unique(np.where(piano_array > 0)[1])
                index = np.where(piano_array > 0)
                song_notes = [0] * (times[len(times)-1]+1)
                for time in times:
                    index_where = np.where(index[1] == time)
                    notes = index[0][index_where][-1]
                    if str(notes) not in self.ntoi:
                        self.ntoi[str(notes)] = self.idx_counter
                        self.iton[self.idx_counter] = str(notes)
                        self.idx_counter += 1
                    song_notes[time] = self.ntoi[str(notes)]
                np.save(self.processed_data +
                        music['midi_name'][5:-4]+'npy', song_notes)

            with open('ntoi.txt', 'w', encoding='utf-8') as outfile:
                json.dump(self.ntoi, outfile)

            with open('iton.txt', 'w', encoding='utf-8') as outfile:
                json.dump(self.iton, outfile)

        elif raw and train_val_test != 'train':
            with open('ntoi.txt', 'r', encoding='utf-8') as infile:
                self.ntoi = json.load(infile)

            with open('iton.txt', 'r', encoding='utf-8') as infile:
                self.iton = json.load(infile)

            for i, music in enumerate(self.info):
                if truncate:
                    if i >= truncate:
                        break
                midi_file = pretty_midi.PrettyMIDI(
                    self.path + music['midi_name'])
                piano_midi = midi_file.instruments[0]
                piano_array = piano_midi.get_piano_roll(fs=30)
                times = np.unique(np.where(piano_array > 0)[1])
                index = np.where(piano_array > 0)
                song_notes = [0] * (times[len(times)-1]+1)
                for time in times:
                    index_where = np.where(index[1] == time)
                    notes = index[0][index_where][-1]
                    if str(notes) not in self.ntoi:
                        song_notes[time] = self.ntoi['UNK']
                    else:
                        song_notes[time] = self.ntoi[str(notes)]
                np.save(self.processed_data +
                        music['midi_name'][5:-4]+'npy', song_notes)

        else:
            with open('ntoi.txt', 'r', encoding='utf-8') as infile:
                self.ntoi = json.load(infile)

            with open('iton.txt', 'r', encoding='utf-8') as infile:
                self.iton = json.load(infile)

        self.dataset = sorted(os.listdir(self.processed_data))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = np.load(self.processed_data + self.dataset[idx])
        encode_data, decode_data, target = self.get_random_seq(item)
        return torch.tensor(encode_data), torch.tensor(decode_data), torch.tensor(target)

    def get_random_seq(self, item):
        index = random.randint(0, len(item)-1)
        if index < self.max_seq:
            encode_data = item[:index-2]
            decode_data = item[index-1]
            encode_data = np.concatenate(
                ([0] * (self.max_seq-len(encode_data)-1), encode_data))
        else:
            encode_data = item[index-1-self.max_seq:index-2]
            decode_data = item[index-1]
        target = item[index]
        return encode_data, decode_data, target


def train(train_loader, model, optimizer, criterion, iterations, device):
    model.train()
    total_loss = 0
    for i in range(iterations):
        for encode_data, decode_data, target in train_loader:
            encode_data, decode_data, target = encode_data.to(
                device), decode_data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(encode_data, decode_data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    return total_loss/len(train_loader)


def eval(val_loader, model, criterion, iterations, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(iterations):
            for encode_data, decode_data, target in val_loader:
                encode_data, decode_data, target = encode_data.to(
                    device), decode_data.to(device), target.to(device)
                output = model(encode_data, decode_data)
                loss = criterion(output, target)
                total_loss += loss.item()
    return total_loss/len(train_loader)


if __name__ == '__main__':
    EPOCHS = 200
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    ITERATIONS = 1

    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    N_LAYERS = 1

    maestro = Maestro(maestro_dir, features_dir, raw=False)
    train_loader = DataLoader(maestro, batch_size=8, shuffle=True)

    validation = Maestro(maestro_dir, features_dir,
                         train_val_test='validation', raw=False)
    val_loader = DataLoader(validation, batch_size=8)

    criterion = nn.NLLLoss()
    model = Seq2Seq(len(maestro.iton), EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE, momentum=MOMENTUM)

    best_loss = sys.maxsize
    early_stop = 0
    early_stop_threshold = 5

    for epoch in range(1, EPOCHS+1):
        train_loss = train(train_loader, model, optimizer,
                           criterion, ITERATIONS, device)
        val_loss = eval(val_loader, model, criterion, ITERATIONS, device)

        print('Epoch {}, Train Loss: {}, Validation Loss: {}'.format(
            epoch, train_loss, val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, output_model_dir+'music-{}.pt'.format(val_loss))
            early_stop = 0

        else:
            early_stop += 1
            if early_stop >= early_stop_threshold:
                print('Early Stopping')
                break
