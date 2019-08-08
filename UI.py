# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:25:48 2019

@author: reube
"""

from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

def file_save(midi):
    f = filedialog.asksaveasfile(mode='w', defaultextension=".txt") #midi
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    # starts from `1.0`, not `0.0`
    f.write(midi)
    f.close() # `()` was missing.
    
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
#file_save("test")

#buttons
btn1 = Button(window, text="Generate!", command = file_save)
btn1.grid(column=2, row = 4, padx=5, pady=5)

window.mainloop()

