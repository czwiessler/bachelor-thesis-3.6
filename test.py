import zipfile

# Path to the downloaded .pth file that may actually be a ZIP file
zip_path = 'C:/Users/christian.zwiessler/.cache/torch/hub/checkpoints/swin_base_patch4_window12_384_22kto1k.pth'

# Check if it is a zip file and list contents
if zipfile.is_zipfile(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Contents of the zipfile:", z.namelist())
else:
    print("The file is not a zip file.")
