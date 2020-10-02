# Ludwig Amadeus
SUTD Artificial Intelligence Project

Authors: Chang Jun Qing, Ho Reuben, Lim Yao Jie, Soong Cun Yuan

## Directory Structure
Directory structure should be as followed:
```
maestro-v2.0.0/
data/ (if not using raw)
checkpoint/
samples/
train.py
models.py
UI.py
generate-large.py
requirements.txt
ntoi.txt (if not using raw)
iton.txt (if not using raw)

```
Maestro 2.0 dataset can be acquired from: `https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip`

## Environment
Using pip:
```
$ pip install -r requirements.txt
```

## Training
```
$ python train.py
```

## Generation
To generate some music, run the following for our GUI
```
$ python UI.py
```

## Large Vocabulary (Additional)
We explored using all possible combinations of notes as well resulting in an ~300k size vocabulary. Run the following
```
$ python generate-large.py --len 100 --notes C4,D4,E4,F4,G4
```
You can also play around with the length of notes and the starting notes
