###########################  Preprocessing of INbreast dataset ###########################
## This script reads the DICOM files from the dataset and saves the final version of    ##
## the images into a png format for latter use.                                         ##
## The dataset is publicly available at :                                               ##
## http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database  ##
##########################################################################################


from PIL import Image
import pydicom
import glob
import numpy as np
import cv2 as cv2

# Setting the dataset directory
thisdir = "/home/gpu-10-2025/Área de trabalho/Datasets/INBreast/DICOM"

images = []; labels = []
tt = 0

for img_path in sorted(glob.glob(thisdir + "/*.dcm")):
    dir1 = img_path.split("/")
    dir2 = dir1[-1]
    dir3 = dir2[:-4]

    ds = pydicom.dcmread(img_path)
    image = ds.pixel_array.astype(np.float32)

    # --- 1) Segmentação simples (como no seu código)
    ret, thresh1 = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    num_labels, labels_im = cv2.connectedComponents(opening.astype(np.uint8))
    values = [np.sum(labels_im == i) for i in range(1, num_labels)]
    big = np.argmax(values) + 1  # maior componente conectado

    final_mask = (labels_im == big)

    # --- 2) Crop do bounding box da mama
    coords = np.column_stack(np.where(final_mask))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    breast = image[y0:y1, x0:x1]
    mask_breast = final_mask[y0:y1, x0:x1]

    # --- 3) Normalizar somente pixels da mama
    fg = breast[mask_breast]

    # evita que zero vire valor válido
    breast_norm = np.full_like(breast, fill_value=-1, dtype=np.float32)

    breast_norm[mask_breast] = (fg - fg.min()) / (fg.max() - fg.min())

    # --- 4) Resize final
    breast_norm = cv2.resize(breast_norm, (256,256), interpolation=cv2.INTER_AREA)

    images.append(breast_norm)
    
    tt = tt +1
    if(tt%100 == 0):
        print('Saving image: {} of 410'.format(tt))
    img = Image.fromarray(np.uint8(breast_norm*255))
    img.save('/home/gpu-10-2025/Área de trabalho/Datasets/INBreast/PNG/' + dir3 + '.png')
