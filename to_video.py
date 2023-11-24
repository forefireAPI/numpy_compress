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

nquant = np.iinfo(np.int16).max

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

    # Création des images pour les octets forts et faibles
    for i in range(input_array.shape[0]):
        high_image = Image.fromarray(high_bytes[i], 'L')
        low_image = Image.fromarray(low_bytes[i], 'L')
        high_image.save(f'temp_images/high_{i:03d}.png')
        low_image.save(f'temp_images/low_{i:03d}.png')

    # Assemblage des images en vidéos avec FFmpeg
    for output_path, image_prefix in zip([output_path_high, output_path_low], ['high', 'low']):
        subprocess.run([
            '/opt/homebrew/bin/ffmpeg', '-y', '-framerate', str(fps), '-i', f'temp_images/{image_prefix}_%03d.png',
    #        '-pix_fmt', 'gray', '-c:v', 'libx265', '-preset', 'medium', 
           #  '-x265-params', 'lossless=1' , '-x265-params', 'strong-intra-smoothing=0:rect=0',     # almost lossless
             '-crf', '10', '-x265-params', 'strong-intra-smoothing=0:rect=0',  # not lossles optimized
           # '-crf', '32 ', # not lossles at all
          #  '-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'gray',
         #   '-metadata', f'min_val={min_val}', '-metadata', f'max_val={max_val}',
            output_path
        ])

    # Nettoyage des images temporaires
    for file in os.listdir('temp_images'):
        os.remove(os.path.join('temp_images', file))
        
    os.rmdir('temp_images')
    return min_val, max_val

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
    V = np.array(array1[time_frame, :, :])
    jhalf = int(nj/2)
    V[:jhalf,:] = array2[time_frame, :jhalf, :]
    V[jhalf,:] = np.max(V)
    
    # Affichage de la trame de array1
    ax1 = plt.subplot(1, 2, 1)
    img1 = ax1.imshow(V, cmap='gray')
    plt.title(f"Semi Image des 2 arrays indice {time_frame}")
    plt.axis('off')
    plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)

    # Affichage de la trame de array2
    ax2 = plt.subplot(1, 2, 2) 
    img2 = ax2.imshow(np.absolute(array2[time_frame, :, :]-array1[time_frame, :, :]), cmap='gray')
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

def check_error_compression(np_array, path_high, path_low):
    """
    Compare la taille en mémoire d'un tableau NumPy avec la somme des tailles des fichiers de deux vidéos.
    
    :param np_array: Tableau NumPy à comparer.
    :param path_high: Chemin du fichier vidéo pour les octets forts.
    :param path_low: Chemin du fichier vidéo pour les octets faibles.
    """
    def get_file_size(file_path):
        """ Renvoie la taille du fichier en octets """
        return os.path.getsize(file_path)

    # Taille du tableau NumPy original en mémoire
    original_size = np_array.nbytes

    # Taille des fichiers vidéo
    size_high = get_file_size(path_high)
    size_low = get_file_size(path_low)
    total_video_size = size_high + size_low

    # Calcul du ratio de compression
    compression_ratio = original_size / total_video_size
    
    min_val = np.min(np_array)
    max_val = np.max(np_array)
    
    array2 = dual_video_to_array(path_high, path_low, min_val, max_val)
    
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
    
    normalized_array = np.int16((np_array - min_val )/ (max_val-min_val) * nquant)
    
    denormalized = (normalized_array.astype(float) / nquant) * (max_val - min_val) + min_val
    
    error_quant_mean = np.sum(np.absolute(np_array-denormalized))/(nk*nj*ni)
    # Affichage des résultats
    print(f"Taille du tableau NumPy original : {original_size} octets")
    print(f"Taille du fichier vidéo (octets forts) : {size_high} octets")
    print(f"Taille du fichier vidéo (octets faibles) : {size_low} octets")
    print(f"Taille totale des fichiers vidéo : {total_video_size} octets")
    print(f"Ratio de compression : {compression_ratio:.2f}")
    print(f"Min : {min_val} Max : {max_val}")
    print(f"Erreur moyenne : {error_mean} valeur erreur mini {error_min}, maxi : {error_max}")
    print(f"Erreur moyenne dequantification : {error_quant_mean}")
    
    plot_arrays(np_array, array2, max_error_frame)
    plot_arrays(np_array, array2, int(nk/2))    
    
    
 
# Paramètres de l'exemple
time_frames = 100  # Nombre de trames temporelles
height, width = 512, 512  # Dimensions spatiales
frequencies = [1, 2, 3,4,5,6,7,8,9]  # Fréquences des ondes sinusoïdales
speeds = [ frequency/10.0 for frequency in frequencies]  # Vitesses de propagation des ondes

# Génération du tableau
#random_array = np.random.rand(time_frames, height, width) # Array Time*Ni*Nj
#array_to_dual_video(random_array,'Routput_high.mp4', 'Routput_low.mp4')

sinusoidal_array = generate_sinusoidal_waves(time_frames, height, width, frequencies, speeds)
array_to_dual_video(sinusoidal_array,'Soutput_high.mp4', 'Soutput_low.mp4')

print("SIN VALUES ")
check_error_compression(sinusoidal_array,'Soutput_high.mp4', 'Soutput_low.mp4')
#print("RANDOM VALUES ")
#check_error_compression(random_array,'Routput_high.mp4', 'Routput_low.mp4')

# array_to_dual_video(input_array)