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

nquant = np.iinfo(np.uint16).max
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
def dual_video_to_array(video_path_high, video_path_low, min_val, max_val ):
    def read_video(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)

        cap.release()
        return np.array(frames) 

    # Lecture des vidéos
    high_bytes  = read_video(video_path_high).astype(np.uint16) 
    low_bytes  = read_video(video_path_low).astype(np.uint16) 
    
    # Vérification de la cohérence des dimensions
    if high_bytes.shape != low_bytes.shape:
        raise ValueError("Les vidéos d'octets forts et faibles ont des dimensions incohérentes")

    # Recombinaison des octets forts et faibles
    combined = (high_bytes << 8) | low_bytes
    
    combined = combined.astype(np.int16) 
    scaled = (combined.astype(float) / nquant) * (max_val - min_val) + min_val
    

    scaled = np.clip(scaled, min_val, max_val)

    # Conversion en XArray
   # xarray_data = xr.DataArray(scaled, dims=["time", "Ni", "Nj"])
    return scaled
 



def array_to_dual_video(input_array, output_path_high , output_path_low , fps=10):
    # Création d'un dossier temporaire pour les images
    if not os.path.exists('temp_images'):
        os.makedirs('temp_images')

    # Normalisation et séparation des octets
    
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    normalized_array = np.int16((input_array - min_val )/ (max_val-min_val) * nquant)
    high_bytes = np.uint8(normalized_array >> 8)
    low_bytes = np.uint8(normalized_array & 0xFF)
    
    #plot_byte_distributions(high_bytes, low_bytes)
 
    # Création des images pour les octets forts et faibles
    for i in range(input_array.shape[0]):
        high_image = Image.fromarray(high_bytes[i], 'L')
        low_image = Image.fromarray(low_bytes[i], 'L')
        high_image.save(f'temp_images/high_{i:03d}.png')
        low_image.save(f'temp_images/low_{i:03d}.png')

    # Assemblage des images en vidéos avec FFmpeg
    for output_path, image_prefix in zip([output_path_high, output_path_low], ['high', 'low']):
        subprocess.run([
            '/opt/homebrew/bin/ffmpeg','-y', '-framerate', str(fps), '-i', f'temp_images/{image_prefix}_%03d.png',
            #'-hide_banner', '-loglevel','warning', '-nostats',
            '-pix_fmt', 'gray', 
            '-c:v', 'libx265', '-preset', 'slow', '-crf', '18',
            #'-x265-params', 'pass=2', 
            #'-x265-params', 'lossless=1' , '-x265-params', 'strong-intra-smoothing=0:rect=0',     # almost lossless
            # '-crf', '10', '-x265-params', 'strong-intra-smoothing=0:rect=0',  # not lossles optimized
           # '-crf', '42 ', # not lossles at all
          #  '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
         #   '-metadata', f'min_val={min_val}', '-metadata', f'max_val={max_val}',
            output_path
        ])

    # Nettoyage des images temporaires
    for file in os.listdir('temp_images'):
        os.remove(os.path.join('temp_images', file))
        
    os.rmdir('temp_images')
    return min_val, max_val
def array_to_video(input_array, output_folder, frame_file='temp_frame.bin'):
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

    # Normalize to 12-bit
    nquant = 2**12 - 1  # 4095 for 12-bit
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

    # Output video file with min and max values encoded in the filename
    output_path = f"{output_folder}/video_min{min_val}_max{max_val}_fc{input_array.shape[0]}.mp4"

    # Define the FFmpeg command
    frame_size = f"{input_array.shape[2]}x{input_array.shape[1]}"  # width x height
    ffmpeg_command = [
        '/opt/homebrew/bin/ffmpeg',
        '-f', 'rawvideo',
        '-pixel_format', 'gray12le',
        '-video_size', frame_size,
        '-framerate', '50',
        '-i', single_frame_file,
        '-c:v', 'libx265',
        '-preset', 'slow', '-crf', '18',
       # '-x265-params', 'lossless=1',
        '-pix_fmt', 'gray12le',
        output_path
    ]

    # Execute the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)

    # Remove the single frame file
   # os.remove(single_frame_file)

    return output_path


def array_to_video_multibin(input_array, output_folder, frame_folder='temp_frames'):
    """
    Convert a 3D numpy array into a 12-bit grayscale video using FFmpeg.
    
    Parameters:
    input_array (numpy.ndarray): The input 3D array.
    output_folder (str): The folder to save the output video file.
    frame_folder (str): Temporary folder to save the intermediate frame files.
    """
    # Ensure the output and frame folders exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(frame_folder, exist_ok=True)

    # Normalize to 12-bit
   
    nquant = 2**12 - 1  # 4095 for 12-bit
    
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    normalized_array = np.int16((input_array - min_val )/ (max_val-min_val) * nquant)
    
 #   plot_byte_distributions(normalized_array, normalized_array,numVals=nquant)
    # Saving 12-bit grayscale images as raw binary files
    for i in range(input_array.shape[0]):
        frame = normalized_array[i].astype(np.uint16)  # Ensure it's 16-bit
        frame_bytes = frame.tobytes()  # Convert to raw bytes

        with open(f'{frame_folder}/frame_{i:05d}.bin', 'wb') as file:
            file.write(frame_bytes)

    # Output video file with min and max values encoded in the filename
    output_path = f"{output_folder}/video_min{min_val}_max{max_val}_fc{input_array.shape[0]}.mp4"

    # Define the FFmpeg command
    frame_size = f"{input_array.shape[2]}x{input_array.shape[1]}"  # width x height
    ffmpeg_command = [
        '/opt/homebrew/bin/ffmpeg',
        '-y',
        '-f', 'image2',
        '-c:v', 'rawvideo',
        '-pix_fmt', 'gray12le',
        '-s:v', frame_size,
        '-r', '50',
        '-i', f'{frame_folder}/frame_%05d.bin',
        '-c:v', 'libx265',
        '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'gray12le',
       # '-x265-params', 'lossless=1',
        output_path
    ]

    # Execute the FFmpeg command
    subprocess.run(ffmpeg_command, check=True)


    # Remove the temporary frame files
    shutil.rmtree(frame_folder)
    return output_path

# Example usage
# input_array = your 3D array
# array_to_video(input_array, 'output_videos')

import re
import glob
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
        frame_count = int(match.group(3))
    else:
        raise ValueError(f"Min, max values, and frame count could not be extracted from the filename: {video_path}")
    
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
        '-i', video_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray16le',
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


def generate_sinusoidal_wavestd_(time_frames, height, width, frequencies, speeds):
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    xv, yv = np.meshgrid(x, y)

    result = np.zeros((time_frames, height, width))
    for t in range(time_frames):
        wave = np.zeros((height, width))
        for freq, speed in zip(frequencies, speeds):
            wave += np.sin(xv * freq + speed * t) + np.sin(yv * freq + speed * t)
        result[t] = wave

    return result

def generate_sinusoidal_waves(time_frames, height, width, frequencies, speeds):
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    xv, yv = np.meshgrid(x, y)

    directions = ['top_to_bottom', 'left_to_right', 'right_to_left', 'diagonal']
    result = np.zeros((time_frames, height, width))

    for t in range(time_frames):
        wave = np.zeros((height, width))
        for freq, speed in zip(frequencies, speeds):
            direction = random.choice(directions)
            if direction == 'top_to_bottom':
                wave += np.sin(yv * freq + speed * t)
            elif direction == 'left_to_right':
                wave += np.sin(xv * freq + speed * t)
            elif direction == 'right_to_left':
                wave += np.sin(-xv * freq + speed * t)
            elif direction == 'diagonal':
                wave += np.sin(xv * freq + yv * freq + speed * t)
            # Add more directions if needed
        result[t] = wave

    return result

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

def check_error_compression(np_array, v_pathes):
    """
    Compare la taille en mémoire d'un tableau NumPy avec la somme des tailles des fichiers de deux vidéos.
    
    :param np_array: Tableau NumPy à comparer.
    :param path_high: Chemin du fichier vidéo pour les octets forts.
    :param path_low: Chemin du fichier vidéo pour les octets faibles.
    """


    # Taille du tableau NumPy original en mémoire
    original_size = np_array.nbytes
    total_video_size = 0
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    array2 = None
    # Taille des fichiers vidéo
    if(len(v_pathes) == 2 ):
        path_high,path_low = v_pathes
        size_high = get_file_size(path_high)
        size_low = get_file_size(path_low)
        total_video_size = size_high + size_low
        array2 = dual_video_to_array(path_high, path_low, min_val, max_val)
    
    else:
        path_12 = v_pathes[0]
        total_video_size = get_file_size(path_12)
        array2 = video_to_array(v_pathes[0])
        print(array2.shape)
        
    compression_ratio = original_size / total_video_size
    
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
    

    plot_error_spectrum(frame_errors )
    nquant = 2**12-1
    normalized_array = np.int16((np_array - min_val )/ (max_val-min_val) * nquant)
    
    denormalized = (normalized_array.astype(float) / nquant) * (max_val - min_val) + min_val
    
    error_quant_mean = np.sum(np.absolute(np_array-denormalized))/(nk*nj*ni)
    # Affichage des résultats
    print(f"Taille du tableau NumPy original : {original_size} octets")
    print(f"Taille totale des fichiers vidéo : {total_video_size} octets")
    print(f"Ratio de compression : {compression_ratio:.2f}")
    print(f"Min : {min_val} Max : {max_val}")
    print(f"Erreur moy: {error_mean} mini {error_min}, max: {error_max}, relMax {error_max/(max_val-min_val)}")
    print(f"Erreur moyenne dequantification : {error_quant_mean}")
    
    plot_arrays(np_array, array2, max_error_frame)
    plot_arrays(np_array, array2, int(nk/2))    
    
    
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

def array_to_grib(np_array, path):    
    from cfgrib.xarray_to_grib import to_grib
    nk,ni,nj = np_array.shape
    for i in range(nk):
        ds2 = xr.DataArray(
             np_array[i,:,:],
             coords=[
                 np.linspace(90., -90., ni),
                 np.linspace(0., 360., nj, endpoint=False),
                 
             ],
             dims=['latitude', 'longitude'],
             ).to_dataset(name='skin_temperature')
        ds2.skin_temperature.attrs['GRIB_shortName'] = 'skt'
        to_grib(ds2, f'{path}/out_{i:05d}.grib')
        
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

in_path = 'video_min-6.076788168243806_max5.021517778572543_fc1301.mp4'
U = video_to_array(in_path)

U_xr = xr.DataArray(U, dims=["time", "nj", "ni"], name='U')
shutil.rmtree('test')
check_error_compression(U,(array_to_video(U,'test'),))


array_to_fgrib(U,"gribAEC.grib")
U_xr.to_dataset(name='U').to_netcdf('U_compressed_fp32.nc', encoding={'U': {'zlib': True, 'dtype': 'f4'}})

np_size = U.nbytes
video_size = get_file_size(in_path)
raw_byte_size =get_file_size('test/temp_frame.bin')
grib_size =  get_file_size("gribAEC.grib")
netcdfz_size = get_file_size('U_compressed_fp32.nc')

print(f"Taille du tableau NumPy original {np_size/np_size}X : {np_size} octets ")
print(f"Taille du fichier vidéo {np_size/video_size}X : {video_size} octets ")
print(f"Taille brute à 2Bytes par valeur {np_size/raw_byte_size}X : {raw_byte_size}")
print(f"Taille grib {np_size/grib_size}X : {grib_size}")
print(f"Taille netcdf fp32 zlib {np_size/netcdfz_size}X : {netcdfz_size}")
