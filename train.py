import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import mido
import csv

maestro_dir = Path('maestro-v2.0.0/')  # maestro-v2.0.0 folder with midi
output_model_dir = Path('checkpoint/')
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Maestro(Dataset):
    '''
    Maestro Dataset
    Inputs:
        path (Path object or str): path to maestro folder
        train_val_test (str): one of 'train', 'validation', or 'test'. Indicates which dataset to use. Defaults: 'train'
    '''

    def __init__(self, path, train_val_test='train'):
        self.path = path
        self.train_val_test = train_val_test
        # Ensures split type is correct
        assert self.train_val_test == 'train' or self.train_val_test == 'validation' or self.train_val_test == 'test'
        self.dataset = []  # holds all the data, data in the form of dictionary with keys: 'composer', 'title', 'year', 'midi_name', 'audio_name', 'duration'
        with open(self.path / 'maestro-v2.0.0.csv', 'r') as f:
            reader = csv.reader(f) # csv reader
            # Omit header line
            first_line = True
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                composer, title, split, year, midi_name, audio_name, duration = line
                if split == self.train_val_test:
                    # Add data into dataset
                    self.dataset.append({'composer': composer, 'title': title, 'year': year,
                                         'midi_name': midi_name, 'audio_name': audio_name, 'duration': duration})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Open midi file with mido
        midi_file = mido.MidiFile(self.path / item['midi_name'])

        '''
        # For reading the whole file out, read https://mido.readthedocs.io/en/latest/midi_files.html to find out more
        for i, track in enumerate(midi_file.tracks):
            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                print(msg)
        '''

        return midi_file
'''
# TEST FUNCTION
maestro = Maestro(maestro_dir)
maestro[5]
'''