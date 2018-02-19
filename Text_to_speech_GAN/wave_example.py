import scipy.io.wavfile as siow 
import scipy.signal as ssr
import matplotlib.pyplot as plt
import math

def groupNumpy(to_group_array, interval):
  '''
  Breaks numpy array into an array of arrays.  Where each array is <interval> long.
  Inputs:
    to_group_array - array we wish to break down into sub intervals
    interval - interval length we wish to break down into 
  Outputs:
    2D array of size [len/interval, interval]
  '''
  print("array lenth: ", len(to_group_array))
  n = math.ceil( len(to_group_array)/(interval*1.0) )
  print("N: ", n)
  return 0

filename = "hello.wav"

audio_tuple = siow.read(filename)
audio_array = audio_tuple[1]
print("Sample rate: ", audio_tuple[0])
#Down sample from 44.1 khz to 16 khz
print("Original Sample len: ", len(audio_array))
audio_new = ssr.resample(audio_array, 16000)

groupNumpy(audio_new, 320)
'''
#Plots look about the same (probably are), so assuming down sampling works for now
plt.plot( audio_new, color="orange" )
plt.ylabel('sound wave')
plt.show()
'''
