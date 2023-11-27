
# README for Python Video Compression and Analysis Tool

## Overview
This Python tool is designed for video compression and analysis of 3D numpy arrays. It focuses on converting numpy arrays into compressed video formats and vice versa, along with providing functions for analyzing the data before and after compression.

## Features
1. **Array-to-Video Conversion:** Converts 3D numpy arrays to 12-bit grayscale videos using FFmpeg.
2. **Video-to-Array Conversion:** Reverts the compressed videos back into numpy arrays.
3. **Data Analysis Tools:** Includes functions for plotting byte distributions, comparing arrays, and visualizing error spectrums.
4. **Support for Multiple Compression Formats:** Various compression settings ranging from very high to lossless compression.
5. **Additional Data Processing Functions:** Includes capabilities like creating a 'super frame' and converting numpy arrays to different formats like GRIB and NetCDF.

## Requirements
- Python 3.x
- Libraries: `cv2`, `numpy`, `matplotlib`, `PIL`, `shutil`, `re`, `xarray`, `cfgrib`
- FFmpeg: Ensure FFmpeg is installed and accessible in the system path.
- Optional: `libx265`, `libopenjpeg` for specific video codecs.

## Installation
1. Install Python 3 and the required Python libraries. You can use pip for this:
   ```bash
   pip install opencv-python numpy matplotlib Pillow shutil regex xarray cfgrib
   ```
2. Install FFmpeg from [FFmpeg's official site](https://ffmpeg.org/download.html) and ensure it's added to your system's PATH.

## Usage
1. **Converting an Array to Video:**
   ```python
   array_to_video(input_array, output_folder, mode)
   ```
   - `input_array`: 3D numpy array to be converted.
   - `output_folder`: Destination folder for the output video.
   - `mode`: Compression mode (e.g., 'high', 'low', etc.).

2. **Converting a Video to an Array:**
   ```python
   video_to_array(video_path)
   ```
   - `video_path`: Path to the input video file.

3. **Analyzing Data:**
   - Use functions like `plot_byte_distributions` and `plot_arrays` for visual analysis.

4. **Data Conversion to GRIB or NetCDF:**
   - Use `array_to_fgrib` for GRIB conversion.
   - NetCDF conversion is demonstrated in the script.

## Result

a compression report such as :

![compression image](https://github.com/forefireAPI/numpy_compress/raw/main/compress.png)


## Example
```python
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

```

## Author
- Jean-Baptiste Filippi

