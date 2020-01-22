import numpy as np
import soundfile as sf
from librosa.feature import mfcc
from keras.utils import to_categorical
import os
import numpy as np
os.system('cls' if os.name == 'nt' else 'clear')

words = ['back', 'forward', 'left', 'right', 'stop']

data = []
label = []
print()
for i in range(1, 81):
	for q in range(0, 25):
		for j in range(0,5):
			f = "in/" + words[j] + "/" + words[j] + str(i) + ".wav"
			wav, samplerate = sf.read(f)
			#wav = np.array(wav)
			x= len(wav)
			#noise = np.random.randn(x)
			#wav = wav + 0.05*noise
			p = 25000-x
			y = (25000//2)-(p//2)#-(q//2)
			new_data = np.zeros([25000,])
			for l in range(0,x):
				new_data[y+l] = wav[l]
			#for q in range(0, 25):
			
			data.append(mfcc(y = new_data, sr = samplerate, n_mfcc=39).T)
			label.append(j)
			print(' ', (len(data)/100), ' %', end='\r')
		

label = np.array(label)
label = to_categorical(label)

data = np.array(data)
np.savez('out/data_cent_nonoise.npz', data=data, label=label)
