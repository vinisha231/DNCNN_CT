0. Setting up the python environment:

To set up the environment to run the DnCnn code, use the following steps:

a) Install conda on your account https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
b) Run the command

conda env create -f dncnnEnv1.yaml --name dncnnEnv1

This creates an environment with all the required python dependencies, including tensorflow.

c) Type "source activate dncnnEnv1"

This activates the environment. You need to do this every time before trying to run the DnCnn code.

1. Generating training data:

Each CT image or sinogram is read in and broken into patches to generate more training data. The patches are then saved all together in a .npy file. There are separate methods to do this for the image-based (postprocessing) and sinogram-based (preprocessing) approaches, to account for the fact that the images are 512 x 512 pixels while the sinograms are 729 x 900 pixels.

a) To generate training data for the image-based corrections (ldct or sparse-view):

python read_float.py --src_dir DnCnnData/images/ndct/train/ --save_dir DnCnnData/images/ndct/train --save_name patches_ndct

This should output:

total patches = 720000 , batch size = 128, total batches = 5625
size of inputs tensor = (720000, 32, 32, 1)

and will create a file called patches_ndct.py in the directory specified under --save_dir. By changing the src_dir, you can generate this for the other ldct or sparse view images.

b) To generate training data for the sinogram-based corrections (ldct only):

python read_float_sino.py --src_dir DnCnnData/sinos/ndct/train --save_dir DnCnnData/sinos/ndct/train --save_name sino_ndct_patches

This should output:

total patches = 842528 , batch size = 128, total batches = 6582
size of inputs tensor = (842528, 27, 36, 1)

and will create a file called sino_ndct_patches.py in the directory specified under --save_dir. 


2. Training network from scratch:

To train the network, run the following code (change paths as required):

python main.py --ndct_training_data "DnCnnData/images/ndct/train/patches_ndct.npy" --ldct_training_data "DnCnnData/images/ldct_1e5/train/patches_ldct.npy" --ndct_eval_data "DnCnnData/images/ndct/test/\*.flt" --ldct_eval_data "DnCnnData/images/ldct_1e5/test/\*.flt"

This specifies the locations of the files containing the image patches (to be used as training data) as well as the test dataset on which the network will be evaluated after every epoch. Some additional parameters can also be specified on the command line, e.g. number of epochs, learning rate, etc.

The program will save "checkpoints" as it runs, storing the best configuration learned so far.

3a. Test data on pretrained network:

To test the data once a network has been trained, use the following command (again, change paths as necessary):  

python main.py --phase test --checkpoint_dir SPIE_results/checkpoints/ldct_1e5_image/ --ndct_test_data "DnCnnData/images/ndct/test/\*.flt" --ldct_test_data "DnCnnData/images/ldct_1e5/test/\*.flt" --test_dir SPIE_results/testoutput/ldct_1e5_image/

The checkpoint_dir should be where the desired checkpoint has been saved, while the ldct test data and ndct test data directories tell the program where to find the test images. Note that only the ldct test data is actually processed by the network; the ndct test data is used to calculate the PSNR. However the PSNR values that are output are not correct (they assume the image is scaled between 0 and 255); we perform the correct calculation in Matlab.

3b. Reconstructing denoised sinogram:

For the sinogram-based denoising, the sinogram must then be reconstructed into an image in order to compare to the NDCT image. This is done by running the file reconstruct_images.m in Matlab. This uses Matlab's built in filtered backprojection routine for fan-beam data (ifanbeam) to reconstruct, and saves the output in .png and .flt format.

4. Analyzing data: 

There are two matlab files that can be used to analyze the data:

a) compare_recons.m: computes the total and average PSNR and SSIM values for all 20 images in one test set. Change file paths at top to run on different datasets.
b) make_figure.m: Used to generate the figures in the 2019 conference paper. Change img_index and the xmin/xmax ymin/ymax values to view different parts of different images (some used for the paper are already present).

