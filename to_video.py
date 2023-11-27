#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:13:21 2023

@author: filippi_j
"""

import cv2
import numpy as np
import subprocess
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil
import re

def plot_byte_distributions(high_bytes, low_bytes, numVals=256):
    """
    Plot the distributions of values in high_bytes and low_bytes side by side.
    
    Parameters:
    high_bytes (numpy.ndarray): The high byte values.
    low_bytes (numpy.ndarray): The low byte values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting the distribution of high_bytes
    axes[0].hist(high_bytes.ravel(), bins=numVals, range=(0, numVals), color='blue', alpha=0.7)
    axes[0].set_title('High Byte Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # Plotting the distribution of low_bytes
    axes[1].hist(low_bytes.ravel(), bins=numVals, range=(0, numVals), color='green', alpha=0.7)
    axes[1].set_title('Low Byte Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def array_to_video(input_array, output_folder, mode , frame_file='temp_frame.bin',ffmpegbin = '/opt/homebrew/bin/ffmpeg'):
    """
    Convert a 3D numpy array into a 12-bit grayscale video using FFmpeg,
    writing all frames to a single binary file.

    Parameters:
    input_array (numpy.ndarray): The input 3D array.
    output_folder (str): The folder to save the output video file.
    frame_file (str): The file to save all the frame data.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    bit_count = mode["bit_count"]
    
    nquant = 2**bit_count - 1  
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    normalized_array = np.int16((input_array - min_val) / (max_val - min_val) * nquant)

    # Path for the single frame file
    single_frame_file = os.path.join(output_folder, frame_file)

    # Write all frames to a single binary file
    with open(single_frame_file, 'wb') as file:
        for i in range(input_array.shape[0]):
            frame = normalized_array[i].astype(np.uint16)  # Ensure it's 16-bit
            frame_bytes = frame.tobytes()  # Convert to raw bytes
            file.write(frame_bytes)

    output_path = f"{output_folder}/video_min{min_val}_max{max_val}_fc{bit_count}.mp4"

    frame_size = f"{input_array.shape[2]}x{input_array.shape[1]}"  # width x height
    
  
    fmpeg_start= [
        ffmpegbin,
        '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo',
        '-pixel_format', 'gray16',
        '-video_size', frame_size,
        '-framerate', '50',
        '-i', single_frame_file,
    ]
    
    ffmpeg_command = fmpeg_start +  mode["ffmpeg_options"] + [output_path]
   # print("Running : "," ".join(ffmpeg_commandXX) )
    print("Expect : "," ".join(ffmpeg_command) )
    subprocess.run(ffmpeg_command, check=True)

    # Remove the single frame file
    # os.remove(single_frame_file)

    return output_path


def video_to_array(video_path, frame_folder='temp_Vframes'):
    """
    Extract frames from a 12-bit grayscale video (stored in 16-bit format) using FFmpeg and load into a numpy array using OpenCV.

    Parameters:
    video_path (str): Path to the input video file.
    frame_folder (str): Temporary folder to save the extracted frame files.

    Returns:
    numpy.ndarray: The reconstructed 3D numpy array.
    """
    # Extract min_val, max_val, and frame_count from the filename
    match = re.search(r"video_min(-?\d+\.\d+)_max(-?\d+\.\d+)_fc(\d+)\.mp4", video_path)
    if match:
        min_val = float(match.group(1))
        max_val = float(match.group(2))
        bit_count = int(match.group(3))
    else:
        raise ValueError(f"Min, max values, and bit count could not be extracted from the filename: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open the video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()


    # Create the frame folder if it doesn't exist
    os.makedirs(frame_folder, exist_ok=True)
    
    raw_frame_file = os.path.join(frame_folder, 'all_frames.raw')

    # Use FFmpeg to extract all frames to a single raw binary file
    ffmpeg_command = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'error',
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray16',
        raw_frame_file
    ]
    subprocess.run(ffmpeg_command, check=True)


    # Read the entire file into a numpy array
    frame_size = frame_width * frame_height * 2  # 2 bytes per pixel for 16-bit grayscale
    total_size = frame_size * frame_count
    
    with open(raw_frame_file, 'rb') as file:
        frame_data = file.read(total_size)
        array = np.frombuffer(frame_data, dtype=np.uint16).reshape((frame_count, frame_height, frame_width))

    # Rescale the data back to its original scale
    array = array.astype(np.float64)
    array = (array / (2**16 - 1)) * (max_val - min_val) + min_val

    # Cleanup: Remove the temporary frame files
    shutil.rmtree(frame_folder)

    return array



def plot_arrays(array1, array2, time_frame):
    """
    Fonction pour afficher deux images côte à côte avec une barre de couleur.

    :param array1: Premier tableau numpy 3D (time_frames, height, width)
    :param array2: Deuxième tableau numpy 3D (time_frames, height, width)
    :param time_frame: Index de la trame temporelle à afficher
    """
    """
    Fonction pour afficher deux images côte à côte.

    :param array1: Premier tableau numpy 3D (time_frames, height, width)
    :param array2: Deuxième tableau numpy 3D (time_frames, height, width)
    :param time_frame: Index de la trame temporelle à afficher
    """
    plt.figure(figsize=(12, 6))
    nk,nj,ni = array2.shape
    V = np.array(array2[time_frame, :, :])
    jhalf = int(nj/2)
    V[:jhalf,:] = array2[time_frame, :jhalf, :]
    V[jhalf,:] = np.max(V)
    
    # Affichage de la trame de array1
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(V, cmap='gray')
    #img1 = ax1.imshow(array1[time_frame, :, :], cmap='gray')
   
    plt.title(f"Semi Image des 2 arrays indice {time_frame}")
    plt.axis('off')
    plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)

    # Affichage de la trame de array2
    ax2 = plt.subplot(1, 2, 2) 
    img2 = ax2.imshow(np.absolute(array2[time_frame, :, :]-array1[time_frame, :, :]), cmap='gray')
    #img2 = ax2.imshow(array2[time_frame, :, :], cmap='gray')
   
    plt.title(f"Difference en valeur absolue des 2 arrays indice {time_frame}")
    plt.axis('off')
    plt.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)


def plot_error_spectrum(frame_errors, num_bins=100):
    """
    Plot the spectrum of max frame errors.

    :param frame_errors: Array of maximum errors for each frame.
    :param bin_size: Size of the bins for quantization.
    """
    # Quantizing the errors
    bins = np.linspace(start=frame_errors.min(), stop=frame_errors.max(), num=num_bins)
    
    plt.hist(frame_errors, bins=bins, edgecolor='black', log=True)

    plt.xlabel('Error Magnitude')
    plt.ylabel('Frequency')
    plt.title('Spectrum of max errors in each step')
    plt.grid(True)
    plt.show()
    
def get_file_size(file_path):
    """ Renvoie la taille du fichier en octets """
    return os.path.getsize(file_path)

def check_error_compression(np_array, np_arrayCOMRESSED, plot=False):


    # Taille du tableau NumPy original en mémoire
    original_size = np_array.nbytes
    total_video_size = 0
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    array2 = np_arrayCOMRESSED
    
    nk,nj,ni = array2.shape
    error_all = np.absolute(np_array-array2)
    error_mean = np.sum(np.absolute(np_array-array2))/(nk*nj*ni) 
    error_max = np.max(error_all)
    error_min = np.min(error_all)

    max_error = -1  # Initial value set to -1 to ensure any real error will be higher
    max_error_frame = -1  # To keep track of the frame with the maximum error
    frame_errors = np.zeros(nk)
    
    for k in range(nk):
        current_frame_error_max = np.max(error_all[k, :, :])
        frame_errors[k] = current_frame_error_max
        if current_frame_error_max > max_error:
            max_error = current_frame_error_max
            max_error_frame = k
    
 
  
 
    print(f"Min : {min_val} Max : {max_val}")
    print(f"Erreur moy: {error_mean} mini {error_min}, max: {error_max}, relMax {error_max/(max_val-min_val)}")
    
    if plot:
        plot_error_spectrum(frame_errors )
        plot_arrays(np_array, array2, max_error_frame)
    return error_mean/(max_val-min_val),error_max,error_max/(max_val-min_val)
 #   plot_arrays(np_array, array2, int(nk/2))    
    
def plot_results(result):
    """
    Plots error metrics and size ratios (both on a logarithmic scale) from the result dictionary.

    :param result: Dictionary containing compression results.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extracting data for plotting
    keys = list(result.keys())
    error_means = [result[key][0] for key in keys]
    error_max = [result[key][1] for key in keys]
    error_rel_max = [result[key][2] for key in keys]
    np_size = result[keys[0]][3]  # Assuming np_size is same for all keys
    video_sizes = [result[key][4] for key in keys]

    # Size comparison with original NumPy array
    np_size_ratios = [np_size / size for size in video_sizes]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting Error Metrics on a logarithmic scale
    ax1.semilogy(keys, error_means, label='Mean Error', marker='o', color='tab:blue')
    ax1.semilogy(keys, error_max, label='Max Error', marker='o', color='tab:orange')
    ax1.semilogy(keys, error_rel_max, label='Relative Max Error', marker='o', color='tab:green')
    ax1.set_ylabel('Error (log scale)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Error Metrics and Size Ratio (Log Scale)')
    ax1.legend(loc='upper left')

    # Creating a twin axis for size ratio (logarithmic scale)
    ax2 = ax1.twinx()
    ax2.bar(keys, np_size_ratios, alpha=0.3, color='tab:red')
    ax2.set_yscale('log')
    ax2.set_ylabel('Compress Ratio (NumPy / Video) - Log Scale', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    
def super_frame(original_array, twoD_shape=( 720,1280)):
    # Create an empty array with the new shape
    nk, _, _ = original_array.shape
    nj, ni = twoD_shape
    new_shape = (nk, nj, ni)
    super_framed_array = np.zeros(new_shape, dtype=original_array.dtype)

    # Define the size of the original patch
    original_patch_size = original_array.shape[1:]

    # Loop through the new array in steps of the original patch size
    for i in range(0, new_shape[1], original_patch_size[0]):
        for j in range(0, new_shape[2], original_patch_size[1]):
            # Determine the type of patch to use based on the position
            if (i // original_patch_size[0]) % 2 == 0:
                if (j // original_patch_size[1]) % 2 == 0:
                    patch = original_array
                else:
                    patch = original_array[:, :, ::-1]  # Horizontal symmetry
            else:
                if (j // original_patch_size[1]) % 2 == 0:
                    patch = original_array[:, ::-1, :]  # Vertical symmetry
                else:
                    patch = original_array[:, ::-1, ::-1]  # Both horizontal and vertical symmetry

            # Adjust the patch size to fit within the bounds of the new array
            end_i = min(i + original_patch_size[0], new_shape[1])
            end_j = min(j + original_patch_size[1], new_shape[2])
            super_framed_array[:, i:end_i, j:end_j] = patch[:, :end_i - i, :end_j - j]

    return super_framed_array


import xarray as xr

        
def array_to_fgrib(np_array, path):    
    from cfgrib.xarray_to_grib import to_grib
    nk,ni,nj = np_array.shape
    ds2 = xr.DataArray(
         np_array,
         coords=[
             np.linspace(0, nk, nk),
             np.linspace(90., -90., ni),
             np.linspace(0., 360., nj, endpoint=False),
             
         ],
         dims=['time','latitude', 'longitude'],
         ).to_dataset(name='skin_temperature')
    ds2.skin_temperature.attrs['gridType'] = 'regular_ll'
    ds2.skin_temperature.attrs['GRIB_shortName'] = 'skt'
    ds2.skin_temperature.attrs['packingType'] ='grid_ccsds'
    to_grib(ds2, path)


c_params = {
    "veryhigh": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '30',
            '-pix_fmt', 'gray16',
        ]
    },
    "high": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'gray16',
        ]
    },
    "normal": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '9',
            '-pix_fmt', 'gray16',
        ]
    },
    "low": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '0',
            '-pix_fmt', 'gray16',
        ]
    },
    "verylow": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-x265-params', 'lossless=1',
            '-pix_fmt', 'gray16',
        ]
    },
    "losslessJP2K": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-map', '0',
            '-c:v', 'libopenjpeg',
            '-c:a', 'copy',
        ]
    },
}



#dataset_path = "/Users/filippi_j/data/2023/prunelli/prunelli15020200809_l0_UVWTKE5000063000.nc"
#ds = xr.open_dataset(dataset_path)
#U = np.sqrt(ds.U.data**2+ds.V.data**2+ds.W.data**2)

in_path = 'video_min-6.076788168243806_max5.021517778572543_fc1301.mp4'
U = video_to_array(in_path)

result = {}
for key in list(c_params.keys())[::1]:
    os.makedirs('test', exist_ok=True)
    shutil.rmtree('test')
    compressed_path = array_to_video(U,'test', c_params[key])
    U_unpacked= video_to_array(compressed_path)            
    error_mean,error_max,error_rel_max = check_error_compression(U,U_unpacked,plot=False)
    np_size = U.nbytes
    video_size = get_file_size(compressed_path)
    raw_byte_size =get_file_size('test/temp_frame.bin')
    result[key] = error_mean,error_max,error_rel_max, np_size,video_size,raw_byte_size
    
for key in result.keys():
    error_mean,error_max,error_rel_max, np_size,video_size,raw_byte_size =result[key] 
    
    print(f"{key} REPORT")
    print(f"{key} Erreur moy: {error_mean:.5f}, max: {error_max:.5f}, relMax {error_rel_max:.5f}")
    print(f"{key} Taille du tableau NumPy original {np_size/np_size:.2f}X : {np_size} octets ")
    print(f"{key} Taille du fichier vidéo {np_size/video_size:.2f}X : {video_size} octets ")
    print(f"{key} Taille brute à 2Bytes par valeur {np_size/raw_byte_size:.2f}X : {raw_byte_size}")

plot_results(result)

check_grib_nc = False
if check_grib_nc:
    array_to_fgrib(U,"gribAEC.grib")
    xr.DataArray(U, dims=["time", "nj", "ni"], name='U').to_dataset(name='U').to_netcdf('U_compressed_fp32.nc', encoding={'U': {'zlib': True, 'dtype': 'f4'}})    
    grib_size =  get_file_size("gribAEC.grib")
    netcdfz_size = get_file_size('U_compressed_fp32.nc')
    print(f"Taille grib {np_size/grib_size:.2f}X : {grib_size}")
    print(f"Taille netcdf fp32 zlib {np_size/netcdfz_size:.2f}X : {netcdfz_size}")




