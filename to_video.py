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
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns


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
        '-framerate', '10',
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

def plot_arrays(array1, array2, time_frame, plot, num_bins=100):
    """
    Display two images side by side, with an error spectrum histogram below the right image.
    The right column (difference image and error spectrum) is smaller than the left image.
    The error spectrum is raised to roughly align with the bottom of the left image.

    :param array1: First 3D numpy array (time_frames, height, width)
    :param array2: Second 3D numpy array (time_frames, height, width)
    :param time_frame: Index of the time frame to display
    :param plot: Identifier or title string used in plot titles and filenames
    :param num_bins: Number of bins for the error histogram (default is 100)
    """


    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    
    # Create a figure with GridSpec: left column larger than right.
    # Adjust height_ratios in the right column so that the error spectrum is compressed.
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 0.5], width_ratios=[2, 1])
    
    # LEFT IMAGE: Combined view from array1 and array2 (spans both rows in left column)
    nk, nj, ni = array2.shape
    V = np.array(array2[time_frame, :, :])
    jhalf = int(nj / 2)
    V[:jhalf, :] = array1[time_frame, jhalf:, :]
    
    ax1 = fig.add_subplot(gs[:, 0])
    img1 = ax1.imshow(V, cmap='turbo')
    ax1.set_title(f"Raw(top)/Compressed - frame: {time_frame}")
    ax1.axis('off')
    fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
    
    # RIGHT TOP: Difference image
    diff = np.abs(array2[time_frame, :, :] - array1[time_frame, :, :])
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = ax2.imshow(diff, cmap='turbo')
    ax2.set_title(f"Config - {plot} Abs difference")
    ax2.axis('off')
    fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
    
    # RIGHT BOTTOM: Error spectrum histogram
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(diff.min(), diff.max(), num_bins)
    ax3.hist(diff.flatten(), bins=bins, edgecolor='black', log=True)
    ax3.set_xlabel('Error Magnitude', fontsize=9)
    ax3.set_ylabel('Frequency', fontsize=9)
    ax3.set_title("Error Spectrum", fontsize=10)
    
    # Adjust tick label sizes to avoid overlaps.
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    
    # Adjust layout: add extra padding to avoid label overlapping.
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{plot}_images.png", dpi=300)

    
def get_file_size(file_path):
    """ Renvoie la taille du fichier en octets """
    return os.path.getsize(file_path)

def check_error_compression(np_array, np_arrayCOMRESSED, plot=None):


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
    
    if plot is not None:
      #  plot_error_spectrum(frame_errors, plot )
        plot_arrays(np_array, array2, 100, plot)
    return error_mean/(max_val-min_val),error_max,error_max/(max_val-min_val)
 #   plot_arrays(np_array, array2, int(nk/2))    
 
def plot_results(result):
    """
    Plots error metrics and size ratios (both on a logarithmic scale) from the result dictionary,
    using a refined style and displaying crisp compression ratios on each bar.

    :param result: Dictionary containing compression results.
    """

    # Use seaborn style for a more refined look
    sns.set_theme(style="whitegrid", context="talk", palette="muted")

    # Extracting data for plotting
    keys = list(result.keys())
    error_means = [result[key]["Errors"][0] for key in keys]
    error_max = [result[key]["Errors"][1] for key in keys]
    error_rel_max = [result[key]["Errors"][2] for key in keys]
    np_size = result[keys[0]]["Errors"][3]  # Assuming np_size is same for all keys
    video_sizes = [result[key]["Errors"][4] for key in keys]

    # Size comparison with original NumPy array
    np_size_ratios = [np_size / size for size in video_sizes]

    # Create the figure and first axis for error metrics
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Plot error metrics on a logarithmic scale
    ax1.semilogy(keys, error_means, label='Mean Error', marker='o', color='tab:blue', linewidth=2)
    ax1.semilogy(keys, error_max, label='Max Error', marker='o', color='tab:orange', linewidth=2)
    ax1.semilogy(keys, error_rel_max, label='Relative Max Error', marker='o', color='tab:green', linewidth=2)
    ax1.set_ylabel('Error (Log Scale)', fontsize=14, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Error Metrics and Compression Ratio', fontsize=16, pad=15)
    ax1.legend(loc='upper right', fontsize=12)

    # Create a twin axis for compression ratio (size ratios)
    ax2 = ax1.twinx()
    bars = ax2.bar(keys, np_size_ratios, alpha=0.5, color='tab:red', width=0.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('Compression Ratio (NumPy / Video) - Log Scale', fontsize=14, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Annotate each bar with the integer compression ratio followed by 'X'
    for bar in bars:
        height = bar.get_height()
        # Convert height to integer compression ratio
        text = f"{round(height, 1):.1f}X"
        ax2.text(bar.get_x() + bar.get_width() / 2, height,
                 text, ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plot.png', dpi=300)

    # Save the result dictionary to JSON
    with open('result.json', 'w') as f:
        json.dump(result, f, indent=4)


    


        
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
    "extreme": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '51',
            '-pix_fmt', 'gray16',
        ]
    },
    "max": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '42',
            '-pix_fmt', 'gray16',
        ]
    },
    "vhigh": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-crf', '28',
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
    "avg": {
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
    "vlow": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-c:v', 'libx265',
            '-preset', 'slow',
            '-x265-params', 'lossless=1',
            '-pix_fmt', 'gray16',
        ]
    },
    "JP2K": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-map', '0',
            '-c:v', 'libopenjpeg',
            '-compression_level', '5',
            '-c:a', 'copy',
        ]
    },
    "llJP2K": {
        "bit_count": 16,
        "ffmpeg_options": [
            '-map', '0',
            '-c:v', 'libopenjpeg',
            '-c:a', 'copy',
        ]
    },
}



dataset_path = "data/V_prunelli_fire.nc"
ds = xr.open_dataset(dataset_path)
print(ds)
#U = np.sqrt(ds.U.data**2+ds.V.data**2+ds.W.data**2)
V =  ds.V.data

#in_path = 'video_min-6.076788168243806_max5.021517778572543_fc1301.mp4'
#U = video_to_array(in_path)
result = {}


for key in list(c_params.keys())[::1]:
    testDir = f"{key}_output"
    os.makedirs(testDir, exist_ok=True)
    shutil.rmtree(testDir)
    compressed_path = array_to_video(V,testDir, c_params[key])
    V_unpacked= video_to_array(compressed_path)
    midFrame = int(np.shape(V_unpacked)[0]/2)

    error_mean,error_max,error_rel_max = check_error_compression(V,V_unpacked,plot=key)
    np_size = V.nbytes
    video_size = get_file_size(compressed_path)
    raw_byte_size =get_file_size(f"{testDir}/temp_frame.bin")
    result[key] = {}
    result[key]["Errors"] = error_mean,error_max,error_rel_max, np_size,video_size,raw_byte_size

check_grib_nc = True
if check_grib_nc:
    array_to_fgrib(V,"gribAEC.grib")
    xr.DataArray(V, dims=["time", "nj", "ni"], name='V') \
  .to_dataset(name='V') \
  .to_netcdf('V_compressed_fp32.nc', encoding={'V': {'zlib': True,
                                                      'complevel': 9,
                                                      'shuffle': True,
                                                      'dtype': 'f4'}})

    xr.DataArray(V, dims=["time", "nj", "ni"], name='V') \
  .to_dataset(name='V') \
  .to_netcdf('V_uncompressed_fp32.nc', encoding={'V': {'dtype': 'f4'}})
    grib_size =  get_file_size("gribAEC.grib")
    netcdfz_size = get_file_size('V_compressed_fp32.nc')
    netcdfu_size = get_file_size('V_uncompressed_fp32.nc')
    
    np_size = V.nbytes
    print(f"Taille grib {np_size/grib_size:.2f}X : {grib_size}")
    print(f"Taille netcdf fp32 zlib {np_size/netcdfz_size:.2f}X : {netcdfz_size}")
    result["GribAEC"] = {}
    result["GribAEC"]["Errors"] = 0,0,0, netcdfu_size,grib_size,np_size
    result["NCZip"] = {}
    result["NCZip"]["Errors"] = 0,0,0, netcdfu_size,netcdfz_size,np_size
    result["NCU"] = {}
    result["NCU"]["Errors"] = 0,0,0, netcdfu_size,netcdfu_size,np_size

for key in result.keys():
    error_mean,error_max,error_rel_max, np_size,video_size,raw_byte_size =result[key]["Errors"] 
    
    print(f"{key} REPORT")
    print(f"{key} Erreur moy: {error_mean:.5f}, max: {error_max:.5f}, relMax {error_rel_max:.5f}")
    print(f"{key} Taille du tableau NumPy original {np_size/np_size:.2f}X : {np_size} octets ")
    print(f"{key} Taille du fichier vidéo {np_size/video_size:.2f}X : {video_size} octets ")
    print(f"{key} Taille brute à 2Bytes par valeur {np_size/raw_byte_size:.2f}X : {raw_byte_size}")
    print(f"{key} Taux de compression effectif : {raw_byte_size/video_size:.2f}")

plot_results(result)



#to show motion vectors : 
#ffmpeg -i input.mp4 -vf "scale=1280:1280" -an output.mp4
#ffmpeg -flags2 +export_mvs -i output.mp4 -vf "split[src],codecview=mv=pf+bf+bb[vex],[vex][src]blend=all_mode=difference128,eq=contrast=7:brightness=-1:gamma=1.5" -c:v libx264 vectors.mp4
#ffmpeg -flags2 +export_mvs -i output.mp4 -vf "split[src],codecview=mv=pf+bf+bb[vex],[vex][src]blend=all_mode=difference128,eq=contrast=7:brightness=0:gamma=1.2" -c:v libx264 vectors.mp4


