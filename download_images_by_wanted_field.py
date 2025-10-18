
import boto3
import os
# DICOM PATH: anon_dicom_path, looks like this:
# /mnt/NAS2/mammo/anon_dicom/cohort_1/66685715/1.2.847.113975.3.62.1.55671517.20170931.1/1.2.844.113684.2750828166.1506503583.4742.253/1.2.826.0.1.3680043.8.498.12113965649920234975762962308241977870.dcm
def download_image_by_name(anon_dicom_path, download_dir = 'downloaded_images', filename_prefix=None):
    """
    Downloads a DICOM image from S3 given its anon_dicom_path from metadata.scv.
    Args:
        anon_dicom_path (str): The name of the DICOM file path (e.g., '/mnt/NAS2/mammo/anon_dicom/cohort_1/66685715/1.2.847.../1.2.844.../1.2.826....dcm').
    
    Preprocessing: in order to match metadata path to S3 bucket actual path structure:

    """
    #  delete the prefix and 'extracted-images' from the path
    anon_dicom_path = anon_dicom_path.replace('/mnt/NAS2/mammo/anon_dicom', 'images') # check that all images have the same prefix
    # folder_prefix = f'png_images/'
    bucket_name = 'embed-dataset-open'
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    dicom_path = f"{anon_dicom_path}" #
    
    # Place each image in a subfolder named after the DICOM file (without extension)
    base_name = os.path.splitext(os.path.basename(anon_dicom_path))[0]
    if filename_prefix is None:
        folder_path = os.path.join(download_dir, base_name)
        filename = os.path.join(folder_path, os.path.basename(anon_dicom_path))
    else:
        folder_path = os.path.join(download_dir, base_name)
        filename = os.path.join(folder_path, filename_prefix + "_" + os.path.basename(anon_dicom_path))
    os.makedirs(folder_path, exist_ok=True)  # <-- Only create the folder

    try:
        bucket.download_file(dicom_path, filename)
        print(f"Downloaded {anon_dicom_path} successfully.")
    except Exception as e:
        print(f"Error downloading {anon_dicom_path}: {e}")
    # recieve dicom path, covert to png and delete dicom
    import pydicom
    import matplotlib.pyplot as plt
    # terminal also "pip install pylibjpeg pylibjpeg-libjpeg"
    # Read the DICOM file
    dicom = pydicom.dcmread(filename)
    
    # Extract the pixel array
    pixel_array = dicom.pixel_array
    
    # Create a PNG file path
    png_pathh = filename.replace('.dcm', '.png')
    
    # Save the pixel array as a PNG image
    plt.imsave(png_pathh, pixel_array, cmap='gray')
    
    # Delete the original DICOM file
    os.remove(filename)
    # print(f"Converted {filename} to {png_pathh} and deleted the DICOM file.")



# download_image_by_name('/mnt/NAS2/mammo/anon_dicom/cohort_1/66685715/1.2.847.113975.3.62.1.55671517.20170931.1/1.2.844.113684.2750828166.1506503583.4742.253/1.2.826.0.1.3680043.8.498.12113965649920234975762962308241977870.dcm')
# download_image_by_name("/mnt/NAS2/mammo/anon_dicom/cohort_1/66685715/1.2.849.113971.3.62.1.57863101.20181005.1/1.2.841.113688.2750828171.1538472340.4875.1595/1.2.826.0.1.3680043.8.498.86259955171961213183997458875961275067.dcm")
# IMAGE BELOW HAS 2 ROI CIRCLES 
# download_image_by_name("/mnt/NAS2/mammo/anon_dicom/cohort_1/83909604/1.2.847.113974.3.63.1.60323819.20191016.1144960/1.2.845.113690.2750824976.1570688042.2516.55173.1/1.2.826.0.1.3680043.8.498.12123416363309366743644254350994480818.dcm")

import pandas as pd
matadata_path = 'C:\\Users\\ayele\\Downloads\\EMBED_OpenData_metadata.csv'
metadata = pd.read_csv(matadata_path)

clinical = pd.read_csv("C:\\emory\\embed\\EMBED_OpenData_clinical.csv")


def create_merged_csv(metadata, clinical, output_path='EMBED_combined_data.csv'):
    """
    CREATE THE "MERGED" CSV from metadata and clinical files
    """
    # print (clinical["path_severity"].value_counts())
    # Merge on patient + exam first
    merged = metadata.merge(
        clinical,
        on=["empi_anon", "acc_anon"],
        suffixes=("_meta", "_clin")
    )

    # Apply laterality matching  
    def laterality_match(row):
        if row["side"] in ["B", None, float("nan")]:  # both or missing
            return True
        return row["side"] == row["ImageLateralityFinal"]

    merged = merged[merged.apply(laterality_match, axis=1)]
    print(merged.shape)
    # Save the merged DataFrame
    merged.to_csv(output_path, index=False)
    return merged

merged = create_merged_csv(metadata, clinical)

# combined = merged, i just did it in colab and downloaded
combined = pd.read_csv("C:\\Users\\ayele\\Downloads\\combined_data_1_180575.csv")
print(combined.head())
# print(combined['asses'].value_counts()) #BIRADS assessment categories
# print(combined['path_severity'].value_counts())

def download_num_images_by_BIRADS(combined, num_images=100):
    # each category of BIRADS "asses" column in EMBED_OpenData_clinical.csv - 0A , 1N, 2B, 3P, 4S, 5M, 6K

    asses_k = combined[combined['asses'] == 'K']['anon_dicom_path'].unique()[:num_images] #2 benign
    asses_b = combined[combined['asses'] == 'B']['anon_dicom_path'].unique()[:num_images] #6 known biopsy proven malignent
    asses_n = combined[combined['asses'] == 'N']['anon_dicom_path'].unique()[:num_images] #1 negative
    asses_p = combined[combined['asses'] == 'P']['anon_dicom_path'].unique()[:num_images] #3 probably benign
    asses_s = combined[combined['asses'] == 'S']['anon_dicom_path'].unique()[:num_images] #4 suspicious
    asses_m = combined[combined['asses'] == 'M']['anon_dicom_path'].unique()[:num_images] #5 highly suggestive of malignency
    asses_a = combined[combined['asses'] == 'A']['anon_dicom_path'].unique()[:num_images] #0 additional evaluation

    asses_k_folder = "C:\\emory\\embed\\downloaded_images_asses\\6K"
    for path in asses_k:
        download_image_by_name(path, download_dir=asses_k_folder)
        # print(f"Downloaded image from path: {path}")

    asses_b_folder = "C:\\emory\\embed\\downloaded_images_asses\\2B"
    for path in asses_b:
        download_image_by_name(path, download_dir=asses_b_folder)
    asses_n_folder = "C:\\emory\\embed\\downloaded_images_asses\\1N"
    for path in asses_n:
        download_image_by_name(path, download_dir=asses_n_folder)

    asses_p_folder = "C:\\emory\\embed\\downloaded_images_asses\\3P"
    for path in asses_p:
        download_image_by_name(path, download_dir=asses_p_folder)

    asses_s_folder = "C:\\emory\\embed\\downloaded_images_asses\\4S"
    for path in asses_s:
        download_image_by_name(path, download_dir=asses_s_folder)
    asses_m_folder = "C:\\emory\\embed\\downloaded_images_asses\\5M"
    for path in asses_m:
        download_image_by_name(path, download_dir=asses_m_folder)
    asses_a_folder = "C:\\emory\\embed\\downloaded_images_asses\\0A"
    for path in asses_a:
        download_image_by_name(path, download_dir=asses_a_folder)

download_num_images_by_BIRADS(combined, num_images=100)

# PATHOLOGY SEVERITY DOWNLOAD
def get_valid_paths(combined, severity):
        # Filter for severity and valid 'asses'
        filtered = combined[
            (combined['path_severity'] == severity) &
            (combined['asses'].notnull()) &
            (combined['asses'] != '') &
            (combined['asses'].astype(str).str.lower() != 'nan')
        ]
        return filtered['anon_dicom_path'].unique()[:200]

def download_by_path_severity():    
    path_sev_0 = get_valid_paths(combined, 0) # 0 invasive cancer
    path_sev_1 = get_valid_paths(combined, 1) # 1 non-invasive cancer
    path_sev_2 = get_valid_paths(combined, 2) # 2 high-risk lesion
    path_sev_3 = get_valid_paths(combined, 3) # 3 borderline lesion
    path_sev_4 = get_valid_paths(combined, 4) # 4 benign findings
    path_sev_5 = get_valid_paths(combined, 5) # 5 normal breast tissue
    # path_sev_6 = combined[combined['path_severity'] == 6]['anon_dicom_path'].unique()[:50] #6 non-breast cancer

    for i in range(6):
        if i == 0 or i == 1:
            continue  # Skip severity levels 0 and 1
        path_list = eval(f'path_sev_{i}')  # Get the actual array of paths
        for path in path_list:
         # find "asses" column value for this path
            match = combined[combined['anon_dicom_path'] == path]
            if not match.empty:
                path_birads = match['asses'].values[0]
            else:
                path_birads = "unknown"
            download_image_by_name(path, download_dir=f"C:\\emory\\embed\\path_severity\\{i}", filename_prefix=path_birads)

download_by_path_severity()

