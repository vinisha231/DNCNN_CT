import argparse
import numpy as np
import struct
from utils import *
from PIL import Image
import glob
import pdb

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='f', default='./noisydata/sparse/train', help='FLT file')
parser.add_argument('--save_dir', dest='save_dir', default='./sparsedata', help='dir of patches')
parser.add_argument('--save_name', dest='save_name',default='patches_ndct', help='name of patch file')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=32, help='patch size')
parser.add_argument('-size', dest='size', default= 512, help='image size')
parser.add_argument('--stride', dest='stride', type=int, default=4, help='stride')
parser.add_argument('--offset', dest='offset', type=int, default=0, help='offset')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
args= parser.parse_args()

AUG_DATA = 1

def makePatches():
    arr= []
    size= args.size   #all images assumed to be the same size
    count= 0
    srcPath = glob.glob(args.f + '/*.flt')
    srcPath= sorted(srcPath)
    #print(srcPath)
    #pdb.set_trace()
    for i in range(len(srcPath)):
        for x in range(0 + args.offset, (size - args.pat_size)+1, args.stride):
            for y in range(0 + args.offset, (size - args.pat_size)+1, args.stride):
                count += 1
    origin_patch_num = count * AUG_DATA
        
    if origin_patch_num % args.bat_size != 0:
       numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
       numPatches = int(numPatches)
    else:
       numPatches = origin_patch_num

    print( "total patches = %d , batch size = %d, total batches = %d" % \
      (numPatches, args.bat_size, numPatches / args.bat_size))
        
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype="f")
    
    count= 0
    for i in range(len(srcPath)):
        arr= np.fromfile(srcPath[i], dtype= '<f')       #matlab binary -> float array
        arr = np.reshape(np.array(arr, dtype="f"),
                         (size, size, 1))               #float array -> np float array
        for j in range(AUG_DATA):
            for x in range(0 + args.offset, size - args.pat_size + 1, args.stride):
                for y in range(0 + args.offset, size - args.pat_size + 1, args.stride):
                    inputs[count, :, :, :] = data_augmentation(arr[x:x + args.pat_size, y:y + args.pat_size, :], \
                          0)
                    count += 1
                
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, args.save_name), inputs)
    print( "size of inputs tensor = " + str(inputs.shape))
    
    exit(0)

if __name__ == '__main__':
    makePatches()       
