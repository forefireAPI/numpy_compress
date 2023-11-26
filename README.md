
# Numpy to video to numpy lossy compression and analysis 

## Overview
This Python script provides a comprehensive solution for compressing numerical data into videos with loss, and subsequently reconstructing and analyzing the compressed data. It is especially useful in scenarios where high-precision numerical data needs to be stored and transmitted efficiently. compares with grib and netcdf.

## Features
- ** Video Compression**: Converts a 3D NumPy array into 12bit videos representing data.
- **Data Reconstruction**: Reconstructs the original NumPy array from the  videos.
- **Error Analysis**: Calculates the average error and quantification error in the compression process.
- **Visualization**: Generates and plots for demonstration and testing purposes.

## Requirements
To run this script, you need the following:
- Python 3
- NumPy
- OpenCV-Python (cv2)
- Pillow (PIL)
- Matplotlib
- FFmpeg (Make sure it's installed and accessible in your system's PATH)

## Installation
1. Ensure Python 3 is installed on your system.
2. Install the required Python libraries: `numpy`, `opencv-python`, `Pillow`, and `matplotlib`.
3. Install FFmpeg and ensure it's accessible from the command line.

## Usage
1. **Video Compression**:
   - Use `array_to_video(input_array, output_path_high, output_path_low)` to compress a 3D NumPy array into dual videos.

2. **Data Reconstruction and Analysis**:
   - Use `check_error_compression(np_array, (path_high)` to reconstruct the NumPy array from the videos and analyze the error.


## Example
```python
in_path = 'video_min-6.076788168243806_max5.021517778572543_fc1301.mp4'
U = video_to_array(in_path)
shutil.rmtree('test')
check_error_compression(U,(array_to_video(U,'test'),))

np_size = U.nbytes
video_size = get_file_size(in_path)
raw_byte_size =get_file_size('test/temp_frame.bin')

print(f"Taille du tableau NumPy original {np_size/np_size}X : {np_size} octets ")
print(f"Taille du fichier vidéo {np_size/video_size}X : {video_size} octets ")
print(f"Taille brute à 2Bytes par valeur {np_size/raw_byte_size}X : {raw_byte_size}")

check_grib_nc = True
if check_grib_nc:
    array_to_fgrib(U,"gribAEC.grib")
    xr.DataArray(U, dims=["time", "nj", "ni"], name='U').to_dataset(name='U').to_netcdf('U_compressed_fp32.nc', encoding={'U': {'zlib': True, 'dtype': 'f4'}})    
    grib_size =  get_file_size("gribAEC.grib")
    netcdfz_size = get_file_size('U_compressed_fp32.nc')
    print(f"Taille grib {np_size/grib_size}X : {grib_size}")
    print(f"Taille netcdf fp32 zlib {np_size/netcdfz_size}X : {netcdfz_size}")
```

## Author
- Jean-Baptiste Filippi

