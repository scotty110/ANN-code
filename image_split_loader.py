import os.path
from multiprocessing import Pool
import sys
import time
import cv2
import numpy as np

def getNewDimensions(width , height , block_size=8):
	""" Since block width = height we can use only one variable to compare"""
	if height % block_size !=0 :
		new_ht = height + (block_size - (height%block_size))
	else:
		new_ht = height
	if width % block_size !=0:
		new_wt = width + (block_size - (width%block_size))
	else:
		new_wt = width
	return new_wt , new_ht

def paddingEdge(input_image_matrix , block_size=8):
  #ht = input_image_matrix.shape[0]
  #wt = input_image_matrix.shape[1]
  ht,wt= input_image_matrix.shape

  if ht % block_size == 0 and wt % block_size == 0:
    #print "Here", type(input_image_matrix)
    #return np.asarray(input_image_matrix)
    return input_image_matrix

  new_width , new_height = getNewDimensions(wt,ht)

  updated_image_matrix = np.zeros(shape=[new_height,new_width] , dtype=np.float64)

  updated_image_matrix[:ht,:wt]=input_image_matrix[:ht,:wt]

  for x in xrange(wt,new_width):
    updated_image_matrix[:ht,x] = updated_image_matrix[:ht,wt-1]
  for y in xrange(ht,new_height):
    updated_image_matrix[y,:new_width] = updated_image_matrix[ht-1,:new_width]
  return updated_image_matrix

def processFile(name, color_channels=0):
  '''
  Process one image, read it in and turn into a matrix
  Inputs:
    name - file name of image we want to read in
    color_channels - 0 if bw, 3 if RGB
  Outputs:
    A matrix representing the image
  '''
  image_matrix = cv2.imread(name,color_channels)
  '''
  ht,wt,channels = image_matrix.shape
  if ht<=wt:
  	dim=(ht,ht)
  else:
  	dim=(wt,wt)
  #dim = (512,512)
  image_matrix = cv2.resize(image_matrix , dim)
  '''
  
  image_matrix = image_matrix.astype(float)
  image_matrix = image_matrix/255.0
  image_matrix = paddingEdge(image_matrix)
  return image_matrix

def processFilesParallel(files):
  ''' Process each file in parallel via Poll.map() '''
  pool=Pool()
  #print "parallel files: ", len(files)
  results=pool.map(processFile, files)
  pool.close()
  pool.join()
  return results

def processDir(dirname, file_ext='.jpg'):
  '''
  Runs parallel inputs
  '''
  files = []
  for file in os.listdir(dirname):
    if file.endswith(file_ext):
        files.append(os.path.join(dirname, file))
  #print "Number of files: ", len(files)
  return files

def getTrainPatches(image_tensor,color=0,patch_size=8):
  '''
  Returns a tenosr of patch tensor from list of images
  Inputs:
    input_files - list of jpeg files to load
    color - 0=BW, 1=RGB
  Output:
    tensor of shape [patches, "time_step", patch_height, patch_width, color_channels]
  '''
  b,m,n = image_tensor.shape   #Leave off color channels
  #Going to assume evenly dividable by 8*8
  row_max = m/patch_size
  col_max = n/patch_size
  p_size = [((row_max-2)*(col_max -2)), 9, b, patch_size, patch_size, color*3]  #We are looking at all 9 surrounding patches
  if color == 0:
    p_size = [((row_max-2)*(col_max -2)), 9, b, patch_size, patch_size]

  patch_array = np.zeros( p_size ) 
  #print "size of patch array: ", (row_max-2)*(col_max-2)
  counter=0
  for i in range(1,row_max-1):
    for j in range(1,col_max-1):
      '''
      This is where I am pulling out the patches from an tensor of 2D images stacked together [image_num,height,width]
      Hard Coded for now
      '''
      #image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j)):(patch_size*(j+1)))][:]
      patch_array[counter,0][:] = image_tensor[:,(patch_size*(i-1)):(patch_size*(i)), (patch_size*(j-1)):(patch_size*(j))][:]
      patch_array[counter,1][:] = image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j-1)):(patch_size*(j))][:]
      patch_array[counter,2][:] = image_tensor[:,(patch_size*(i+1)):(patch_size*(i+2)), (patch_size*(j-1)):(patch_size*(j))][:]
      patch_array[counter,3][:] = image_tensor[:,(patch_size*(i-1)):(patch_size*(i)), (patch_size*(j)):(patch_size*(j+1))][:]
      patch_array[counter,4][:] = image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j)):(patch_size*(j+1))][:]
      patch_array[counter,5][:] = image_tensor[:,(patch_size*(i+1)):(patch_size*(i+2)), (patch_size*(j)):(patch_size*(j+1))][:]
      patch_array[counter,6][:] = image_tensor[:,(patch_size*(i-1)):(patch_size*(i)), (patch_size*(j+1)):(patch_size*(j+2))][:]
      patch_array[counter,7][:] = image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j+1)):(patch_size*(j+2))][:]
      patch_array[counter,8][:] = image_tensor[:,(patch_size*(i+1)):(patch_size*(i+2)), (patch_size*(j+1)):(patch_size*(j+2))][:]
      counter=counter+1
  #Get inner patches to train on
  del image_tensor
  if color == 0:
    patch_array = np.transpose(patch_array, (0, 2, 1, 3, 4))
  else:
    patch_array = np.transpose(patch_array, (0, 2, 1, 3, 4, 5))
  return patch_array

def getPredictionPatches(image_tensor,color=0,patch_size=8):
  '''
  Returns a tenosr of patch tensor from list of images
  Inputs:
    input_files - list of jpeg files to load
    color - 0=BW, 1=RGB
  Output:
    tensor of shape [patches,patch_height, patch_width, color_channels]
  '''
  b,m,n = image_tensor.shape   #Leave off color channels
  #Going to assume evenly dividable by 8*8
  row_max = m/patch_size
  col_max = n/patch_size
  p_size = [((row_max-2)*(col_max -2)), b, patch_size, patch_size, color*3]  #We are looking at all 9 surrounding patches
  if color == 0:
    p_size = [((row_max-2)*(col_max -2)), b, patch_size, patch_size]

  patch_array = np.zeros( p_size ) 
  #print "size of patch array: ", (row_max-2)*(col_max-2)
  counter=0
  for i in range(1,row_max-1):
    for j in range(1,col_max-1):
      #image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j)):(patch_size*(j+1)))][:]
      patch_array[counter][:] = image_tensor[:,(patch_size*(i)):(patch_size*(i+1)), (patch_size*(j)):(patch_size*(j+1))][:]
      counter=counter+1
  #Get inner patches to train on
  del image_tensor
  return patch_array

def nextbatch(batch_size, comp_file_array, raw_file_array, starting_point):
  '''
  Returns the stack of images in correct batch size
  Inputs:
    batch_size - batch size wish to train on
    file_array - array of file names that we wish to train on
    starting_point - which file we wish to start at
  Outputs:
    An array of correct batch size and next starting position
  '''
  end_pos = starting_point+batch_size
  
  #Get prediction patches
  image_files_r = raw_file_array[starting_point:end_pos]
  image_matrix_r = processFilesParallel(image_files_r)
  image_matrix_r = np.asarray(image_matrix_r)
  '''
  if type(image_files_r) is not np.ndarray:
    image_matrix_r = np.asarray(image_matrix_r)
  '''
  patches_r = getPredictionPatches(image_matrix_r,color=0)
  
  #get raw patches
  image_files_c = comp_file_array[starting_point:end_pos]
  image_matrix_c = processFilesParallel(image_files_c)
  image_matrix_c = np.asarray(image_matrix_c)
  '''
  if type(image_files_c) is not np.ndarray:
    image_matrix_c = np.asarray(image_matrix_c)
  '''
  patches_c = getTrainPatches(image_matrix_c,color=0)

  if end_pos > len(raw_file_array):
    end_pos = 0
  return patches_c, patches_r, end_pos

if __name__ == '__main__':
  input_files = "" <TODO>
  in_list = processDir(input_files)
  t_1, t_2, count = nextbatch(2,in_list,in_list,0)
  
  print "Tester shape: ", t_1.shape
