import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pretty_midi
import csv
import numpy as np
import random
import json


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

        if raw:
            self.ntoi = {0: 0}  # Dictionary to map the notes
            self.iton = {0: 0}

            self.idx_counter = 1  # Counter for notes mapping

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
                    notes = tuple(index[0][index_where])
                    if notes not in self.ntoi:
                        self.ntoi[notes] = self.idx_counter
                        self.iton[self.idx_counter] = notes
                        self.idx_counter += 1
                    song_notes[time] = self.ntoi[notes]
                np.save(self.processed_data +
                        music['midi_name'][5:-4]+'npy', song_notes)

            with open('ntoi.txt', 'w', encoding='utf-8') as outfile:
                json.dump(self.ntoi, outfile)

            with open('iton.txt', 'w', encoding='utf-8') as outfile:
                json.dump(self.iton, outfile)

        self.dataset = os.listdir(self.processed_data)
        print(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        data, target = self.get_random_seq(item)
        return torch.tensor([data]), torch.tensor([target])

    def get_random_seq(self, item):
        assert len(item) > self.max_seq
        index = random.randint(0, len(item)-self.max_seq-1)
        data = item[index:index+self.max_seq]
        target = item[index+1:index+self.max_seq+1]
        return data, target


if __name__ == '__main__':

    # TEST FUNCTION
    maestro = Maestro(maestro_dir, features_dir, raw=True)
    # data, target = maestro[9]
    # print(data.shape)
    # print(target.shape)
