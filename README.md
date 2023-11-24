
# Dual Video Compression and Analysis Tool

## Overview
This Python script provides a comprehensive solution for compressing numerical data into dual videos with high and low bytes, and subsequently reconstructing and analyzing the compressed data. It is especially useful in scenarios where high-precision numerical data needs to be stored and transmitted efficiently.

## Features
- **Dual Video Compression**: Converts a 3D NumPy array into two separate videos representing high and low bytes of data.
- **Data Reconstruction**: Reconstructs the original NumPy array from the dual videos.
- **Error Analysis**: Calculates the average error and quantification error in the compression process.
- **Visualization**: Generates and plots sinusoidal wave patterns for demonstration and testing purposes.

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
1. **Dual Video Compression**:
   - Use `array_to_dual_video(input_array, output_path_high, output_path_low)` to compress a 3D NumPy array into dual videos.

2. **Data Reconstruction and Analysis**:
   - Use `check_error_compression(np_array, path_high, path_low)` to reconstruct the NumPy array from the dual videos and analyze the error.

3. **Visualization**:
   - Use `generate_sinusoidal_waves` to create a sinusoidal wave pattern for testing.
   - The script also contains functions to plot arrays and compare them visually.

## Example
```python
# Generate a sinusoidal wave pattern
sinusoidal_array = generate_sinusoidal_waves(time_frames, height, width, frequencies, speeds)

# Compress the generated array into dual videos
array_to_dual_video(sinusoidal_array, 'Soutput_high.mp4', 'Soutput_low.mp4')

# Analyze the compression and reconstruct the data
print("SIN VALUES")
check_error_compression(sinusoidal_array, 'Soutput_high.mp4', 'Soutput_low.mp4')
```

## Author
- Jean-Baptiste Filippi

