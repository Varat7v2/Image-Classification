import cv2
import os, sys, glob
from tqdm import tqdm
import csv


PROJECT_NAME = 'gender_detection'
PARENT_DATASET_PATH = 'data_gender'
DESTINATION_DATASET_PATH = 'gender_dataset'

data = list()

classes = ['female', 'male']
if not os.path.exists(DESTINATION_DATASET_PATH):
    os.makedirs(DESTINATION_DATASET_PATH)

for iclass in classes:
	if not os.path.exists(os.path.join(DESTINATION_DATASET_PATH, iclass)):
	    os.makedirs(os.path.join(DESTINATION_DATASET_PATH, iclass))

for mydir in tqdm(os.listdir(PARENT_DATASET_PATH)):
	count = 1
	for subdir in os.listdir(os.path.join(PARENT_DATASET_PATH, mydir)):
		for img_path in glob.glob(os.path.join(PARENT_DATASET_PATH,mydir,subdir+'/*')):
			data.append([img_path])
			frame = cv2.imread(img_path)
			cv2.imwrite(os.path.join(DESTINATION_DATASET_PATH,mydir)+'/'+str(count)+'.jpg', frame)
			count += 1

file = open(PROJECT_NAME+'.csv', 'w+', newline='')

with file:
	writer = csv.writer(file)
	writer.writerows(data)