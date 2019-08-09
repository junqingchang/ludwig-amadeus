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

def write_midi_file_from_generated(generate, midi_file_name = "result.mid", start_index=49, fs=8, max_generated=1000):
  note_string = [pretty_midi.note_number_to_name(ind_note) for ind_note in generate]
  array_piano_roll = np.zeros((128,max_generated+1), dtype=np.int16)
  for index, note in enumerate(note_string[start_index:]):
    if note == 'e':
      pass
    else:
      splitted_note = note.split(',')
      for j in splitted_note:
        array_piano_roll[int(j),index] = 1
  generate_to_midi = pretty_midi.piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
  print("Tempo {}".format(generate_to_midi.estimate_tempo()))
  for note in generate_to_midi.instruments[0].notes:
    note.velocity = 100
  generate_to_midi.write(midi_file_name)

def gen_music():
    #gen music here
    #file_save("music")
    note = comboNote.get()+comboOctave.get()
    noteNo = pretty_midi.note_name_to_number(note)
#    print(note,noteNo,pretty_midi.note_number_to_name(noteNo))
    generate = generate_from_one_note(noteNo)
    
def generate_from_one_note(note='60'):
  generate = ['0' for i in range(49)]
  generate += [note]
  return generate

#def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=50):
#  for i in tqdm_notebook(range(max_generated), desc='genrt'):
#    test_input = np.array([generate])[:,i:i+seq_len]
#    predicted_note = model.predict(test_input)
#    random_note_pred = choice(unique_notes+1, 1, replace=False, p=predicted_note[0])
#    generate.append(random_note_pred[0])
#  return generate

   
window = Tk()
window.title("Ludwig Amadeus")
window.geometry("400x150")

#labels
lbl1 = Label(window, wraplength=400, font=("Arial Bold",18), text = "Generate your very own music by setting the initial values below!")
lbl1.grid(row=0, column =0, columnspan = 3)
Label(window, text="First Note").grid(row=1, sticky=W)
Label(window, text="Octave").grid(row=2, sticky=W)

comboNote = Combobox(window)
comboNote['values']= ('C','D','E','F','G','A','B')
comboNote.current(0)
comboNote.grid(column=1, row = 1)

comboOctave = Combobox(window)
comboOctave['values']= (0,1,2,3,4,5,6,7,8)
comboOctave.current(4)
comboOctave.grid(column=1, row = 2)


#buttons
btn1 = Button(window, text="Generate!", command = gen_music)
btn1.grid(column=2, row = 4, padx=5, pady=5)

window.mainloop()

