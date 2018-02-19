import scipy.io.wavfile as siow 
import scipy.signal as ssr
import matplotlib.pyplot as plt
import numpy as np
import math

def groupNumpy(to_group_array, interval, debug=True):
  '''
  Breaks numpy array into an array of arrays.  Where each array is <interval> long.
  Inputs:
    to_group_array - array we wish to break down into sub intervals
    interval - interval length we wish to break down into 
  Outputs:
    2D array of size [len/interval, interval]
  '''
  
  n = math.ceil( len(to_group_array)/(interval*1.0) )
  if(debug):
    print("array lenth: ", len(to_group_array))
    print("N: ", n)
  
  # Make 2D array 
  output_array = np.zeros((n,interval,2))
  
  for i in range(0,n,1):
    output_array[i] = to_group_array[(i*interval):((i+1)*interval)]
  
  return output_array

filename = "hello.wav"

audio_tuple = siow.read(filename)
audio_array = audio_tuple[1]
print("Sample rate: ", audio_tuple[0])

#Assuming sample rate is 16 khz, want to break into 20 ms chucks
grouped_array = groupNumpy(audio_array, 320)

#print("differenc: ",  grouped_array[1])
#print("true segment: ", audio_array[320:2*320])
print("Difference: ", np.sum(grouped_array[1]-audio_array[320:2*320]) )
#2D sound is multi channel sound

'''
#Plots look about the same (probably are), so assuming down sampling works for now
plt.plot( audio_new, color="orange" )
plt.ylabel('sound wave')
plt.show()
'''


'''
Used as a outline for code:
https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a
'''