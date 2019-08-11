# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:25:48 2019

@author: reube
"""

from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *
import pretty_midi
import numpy as np
import torch
import json
from torch.distributions.categorical import Categorical
import random
import copy
import operator

MODEL_PATH = "music.pt"
ntoi_path = "ntoi.txt"
iton_path = "iton.txt"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

with open(ntoi_path) as f:
    ntoi = json.loads(f.read())

with open(iton_path) as f:
    iton = json.loads(f.read())
    
def write_midi_file_from_generated():
    f = filedialog.asksaveasfile(mode='w', defaultextension=".mid")
    if f:
        fname = f.name
        f.close()
        # Create a PrettyMIDI object
    piano_chord = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a cello instrument
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    generate = gen_music()
    start = 0
    for note in generate:
        note_number = int(iton[str(note)])
        duration = random.uniform(0.15,0.45)
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start, end=start+duration)
        piano.notes.append(note)
        start+=duration
        
    # Add the piano instrument to the PrettyMIDI object
    piano_chord.instruments.append(piano)
    # Write out the MIDI data
    piano_chord.write(fname)

def gen_music():
    #gen music here
    #file_save("music")
    model = torch.load(MODEL_PATH)
    model.eval()
    model.to(device)
    notes = []
    note = comboNote1.get()
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[noteNo])
    note = comboNote2.get()
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[noteNo])
    note = comboNote3.get()
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[noteNo])
    note = comboNote4.get()
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[noteNo])
    note = comboNote5.get()
    noteNo = str(pretty_midi.note_name_to_number(note))
    notes.append(ntoi[noteNo])
    generate = generate_from_one_note(notes)
    
    for i in range(195):
        output = torch.exp(model(torch.tensor([generate]).to(device)))
        newarr = [i[0] for i in output[0]]
        if tobeam.get():
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
  generate = [0 for i in range(195)]
  for i in notes:
      generate += [i]
  return generate
   
window = Tk()
window.title("Ludwig Amadeus")
window.geometry("390x250")
notes=[]
for i in [0,1,2,3,4,5,6,7,8]:
    for j in ['C','D','E','F','G','A','B']:
        notes.append(j+str(i))
#labels
lbl1 = Label(window, wraplength=400, font=("Arial Bold",18), text = "Generate your very own music by setting the initial values below!")
lbl1.grid(row=0, column =0, columnspan = 3)
Label(window, text="First Note").grid(row=1, sticky=W)
Label(window, text="Second Note").grid(row=2, sticky=W)
Label(window, text="Third Note").grid(row=3, sticky=W)
Label(window, text="Fourth Note").grid(row=4, sticky=W)
Label(window, text="Fifth Note").grid(row=5, sticky=W)

comboNote1 = Combobox(window)
comboNote1['values']= notes
comboNote1.current(28)
comboNote1.grid(column=1, row = 1, padx=3, pady=3)

comboNote2 = Combobox(window)
comboNote2['values']= notes
comboNote2.current(28)
comboNote2.grid(column=1, row = 2, padx=3, pady=3)

comboNote3 = Combobox(window)
comboNote3['values']= notes
comboNote3.current(28)
comboNote3.grid(column=1, row = 3, padx=3, pady=3)

comboNote4 = Combobox(window)
comboNote4['values']= notes
comboNote4.current(28)
comboNote4.grid(column=1, row = 4, padx=3, pady=3)

comboNote5 = Combobox(window)
comboNote5['values']= notes
comboNote5.current(28)
comboNote5.grid(column=1, row = 5, padx=3, pady=3)

tobeam = IntVar()
Checkbutton(window, text="Beam search", variable=tobeam).grid(row=6, sticky=W)
#buttons
btn1 = Button(window, text="Generate!", command = write_midi_file_from_generated)
btn1.grid(column=2, row = 7, padx=5, pady=5)

window.mainloop()

