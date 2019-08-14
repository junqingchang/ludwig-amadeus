import pretty_midi
import numpy as np
import torch
import json
from torch.distributions.categorical import Categorical
import random
import copy
import operator
import tqdm

import argparse

parser = argparse.ArgumentParser(description='Create Music')
parser.add_argument('--len', default=100, type=int, help='number of notes to generate')
parser.add_argument('--notes', default='C4,E4,G4,F4,E4', type=str, help='starting 5 notes, e.g. C4,E4,G4,F4,E4')
parser.add_argument('--output', default='demo.mid', type=str, help='filename ending with .mid')

args = parser.parse_args()

MODEL_PATH = "checkpoint/300k-music.pt"
ntoi_path = "ntoi-300k.txt"
iton_path = "iton-300k.txt"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

output = args.output

starting_notes = args.notes.split(',')
do_beam = False
song_len = args.len

with open(ntoi_path) as f:
    ntoi = json.loads(f.read())

with open(iton_path) as f:
    iton = json.loads(f.read())


def write_midi_file_from_generated(filename):
        # Create a PrettyMIDI object
    piano_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    generate = gen_music()
    start = 0
    for note in generate:
        ref = iton[str(note)]
        if ref != 0:
            notes = [int(x) for x in iton[str(note)][1:-1].split(',') if x != '']
        else:
            notes = [0]
        duration = random.uniform(0.15, 0.45)        
        for n in notes:
            note = pretty_midi.Note(
                velocity=100, pitch=n, start=start, end=start+duration)
            piano.notes.append(note)
        start += duration

    # Add the piano instrument to the PrettyMIDI object
    piano_chord.instruments.append(piano)
    # Write out the MIDI data
    piano_chord.write(filename)


def gen_music():
    # gen music here
    # file_save("music")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)
    notes = []
    note = starting_notes[0]
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[f'({noteNo},)'])
    note = starting_notes[1]
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[f'({noteNo},)'])
    note = starting_notes[2]
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[f'({noteNo},)'])
    note = starting_notes[3]
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[f'({noteNo},)'])
    note = starting_notes[4]
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[f'({noteNo},)'])
    generate = generate_from_one_note(notes)

    for i in tqdm.tqdm(range(song_len-5)):
        output = torch.exp(model(torch.tensor([generate]).to(device)))
        newarr = [i[0] for i in output[0]]
        if do_beam:
            beam = {}
            for j in range(3):
                c = Categorical(torch.tensor(newarr))
                next_note = c.sample()
                beam[next_note.item()] = newarr[next_note.item()]
            for k in beam:
                arr = copy.deepcopy(generate)
                arr.pop(0)
                arr.append(k)
                instance_out = torch.exp(model(torch.tensor([arr]).to(device)))
                instarr = [i[0] for i in instance_out[0]]
                c = Categorical(torch.tensor(instarr))
                next_note = c.sample()
                beam[k] = beam[k]*instarr[next_note.item()]
            generate.pop(0)
            generate.append(max(beam.items(), key=operator.itemgetter(1))[0])
        else:
            c = Categorical(torch.tensor(newarr))
            next_note = c.sample()
            generate.pop(0)
            generate.append(next_note.item())
    return generate


def generate_from_one_note(notes):
    generate = [0 for i in range(song_len-5)]
    for i in notes:
        generate += [i]
    return generate

if __name__ == '__main__':
    write_midi_file_from_generated(output)