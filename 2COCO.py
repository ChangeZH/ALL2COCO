import os
import cv2
import json
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

COCO_DICT=['images','annotations','categories']

IMAGES_DICT=['file_name','height','width','id']
## {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
## 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
## 'flickr_url': '', 'id': 516316}

ANNOTATIONS_DICT=['image_id','ifcrowd','bbox','category_id','id']
## {'segmentation': [[]],
## 'area': 58488.148799999995, 'iscrowd': 0, 
## 'image_id': 170893, 'bbox': [270.55, 80.55, 367.51, 393.7],
## 'category_id': 18, 'id': 9940}

CATEGORIES_DICT=['id','name']
## {'supercategory': 'person', 'id': 1, 'name': 'person'}
## {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}

RSOD_CATEGORIES=['aircraft','playground','overpass','oiltank']
NWPU_CATEGORIES=['airplane','ship','storage tank','baseball diamond','tennis court',\
					'basketball court','ground track field','harbor','bridge','vehicle']
VOC_CATEGORIES=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',\
					'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
DIOR_CATEGORIES=['golffield','Expressway-toll-station','vehicle','trainstation','chimney','storagetank',\
					'ship','harbor','airplane','groundtrackfield','tenniscourt','dam','basketballcourt',\
					'Expressway-Service-area','stadium','airport','baseballfield','bridge','windmill','overpass']

parser=argparse.ArgumentParser(description='2COCO')
parser.add_argument('--image_path',type=str,default='./aircraft/JPEGImages/',help='config file')
parser.add_argument('--annotation_path',type=str,default='./aircraft/Annotation/labels/',help='config file')
parser.add_argument('--dataset',type=str,default='RSOD',help='config file')
parser.add_argument('--save',type=str,default='./test.json',help='config file')

args=parser.parse_args()

def load_json(path):
	with open(path,'r') as f:
		json_dict=json.load(f)
		for i in json_dict:
			print(i)
		print(json_dict['annotations'])

def save_json(dict,path):
	print('SAVE_JSON...')
	with open(path,'w') as f:
		json.dump(dict,f)
	print('SUCCESSFUL_SAVE_JSON:',path)

def load_image(path):
	img=cv2.imread(path)
	return img.shape[0],img.shape[1]

def generate_categories_dict(category):
	print('GENERATE_CATEGORIES_DICT...')
	return [{CATEGORIES_DICT[0]:category.index(x),CATEGORIES_DICT[1]:x} for x in category]

def generate_images_dict(imagelist,image_path,start_image_id):
	print('GENERATE_IMAGES_DICT...')
	images_dict=[]
	with tqdm(total=len(imagelist)) as load_bar:
		for x in imagelist:
			dict={IMAGES_DICT[0]:x,IMAGES_DICT[1]:load_image(image_path+x)[0],\
					IMAGES_DICT[2]:load_image(image_path+x)[1],IMAGES_DICT[3]:imagelist.index(x)+start_image_id}
			load_bar.update(1)
			images_dict.append(dict)
	return images_dict
	# return [{IMAGES_DICT[0]:x,IMAGES_DICT[1]:load_image(image_path+x)[0],\
	# 				IMAGES_DICT[2]:load_image(image_path+x)[1],IMAGES_DICT[3]:imagelist.index(x)+start_image_id} for x in imagelist]

def NWPU_Dataset(image_path,annotation_path,start_image_id=0,start_id=0):

	categories_dict=generate_categories_dict(NWPU_CATEGORIES)

	imgname=os.listdir(image_path)
	images_dict=generate_images_dict(imgname,image_path,start_image_id)

	print('GENERATE_ANNOTATIONS_DICT...')
	annotations_dict=[]
	for i in images_dict:
		image_id=i['id']
		image_name=i['file_name']
		annotation_txt=annotation_path+image_name.split('.')[0]+'.txt'

		txt=open(annotation_txt,'r')
		lines=txt.readlines()
		id=start_id
		for j in lines:
			if j=='\n':
				continue
			category_id=int(j.split(',')[4])-1
			category=NWPU_CATEGORIES[category_id]
			x_min=float(j.split(',')[0].split('(')[1])
			y_min=float(j.split(',')[1].split(')')[0])
			w=float(j.split(',')[2].split('(')[1])-x_min
			h=float(j.split(',')[3].split(')')[0])-y_min
			bbox=[x_min,y_min,w,h]
			dict={'image_id':image_id,'ifcrowd':0,'bbox':bbox,'category_id':category_id,'id':id}
			id=id+1
			annotations_dict.append(dict)
	print('SUCCESSFUL_GENERATE_NWPU_JSON')
	return {COCO_DICT[0]:images_dict,COCO_DICT[1]:annotations_dict,COCO_DICT[2]:categories_dict}

def RSOD_Dataset(image_path,annotation_path,start_image_id=0,start_id=0):

	categories_dict=generate_categories_dict(RSOD_CATEGORIES)

	imgname=os.listdir(image_path)
	images_dict=generate_images_dict(imgname,image_path,start_image_id)

	print('GENERATE_ANNOTATIONS_DICT...')
	annotations_dict=[]
	for i in images_dict:
		image_id=i['id']
		image_name=i['file_name']
		annotation_txt=annotation_path+image_name.split('.')[0]+'.txt'

		txt=open(annotation_txt,'r')
		lines=txt.readlines()
		id=start_id
		for j in lines:
			category=j.split('\t')[1]
			category_id=RSOD_CATEGORIES.index(category)
			x_min=float(j.split('\t')[2])
			y_min=float(j.split('\t')[3])
			w=float(j.split('\t')[4])-x_min
			h=float(j.split('\t')[5])-y_min
			bbox=[x_min,y_min,w,h]
			dict={'image_id':image_id,'ifcrowd':0,'bbox':bbox,'category_id':category_id,'id':id}
			annotations_dict.append(dict)
			id=id+1
	print('SUCCESSFUL_GENERATE_RSOD_JSON')
	return {COCO_DICT[0]:images_dict,COCO_DICT[1]:annotations_dict,COCO_DICT[2]:categories_dict}

def  DIOR_Dataset(image_path,annotation_path,start_image_id=0,start_id=0):

	categories_dict=generate_categories_dict(DIOR_CATEGORIES)

	imgname=os.listdir(image_path)
	images_dict=generate_images_dict(imgname,image_path,start_image_id)

	print('GENERATE_ANNOTATIONS_DICT...')
	annotations_dict=[]
	for i in images_dict:
		image_id=i['id']
		image_name=i['file_name']
		annotation_xml=annotation_path+image_name.split('.')[0]+'.xml'

		tree=ET.parse(annotation_xml)
		root=tree.getroot()

		id=start_id
		for j in root.findall('object'):
			category=j.find('name').text
			category_id=DIOR_CATEGORIES.index(category)
			x_min=float(j.find('bndbox').find('xmin').text)
			y_min=float(j.find('bndbox').find('ymin').text)
			w=float(j.find('bndbox').find('xmax').text)-x_min
			h=float(j.find('bndbox').find('ymax').text)-y_min
			bbox=[x_min,y_min,w,h]
			dict={'image_id':image_id,'ifcrowd':0,'bbox':bbox,'category_id':category_id,'id':id}
			annotations_dict.append(dict)
			id=id+1
	print('SUCCESSFUL_GENERATE_DIOR_JSON')
	return {COCO_DICT[0]:images_dict,COCO_DICT[1]:annotations_dict,COCO_DICT[2]:categories_dict}

if __name__=='__main__':

	dataset=args.dataset
	save=args.save
	image_path=args.image_path
	annotation_path=args.annotation_path

	if dataset=='RSOD':
		json_dict=RSOD_Dataset(image_path,annotation_path,0)
		save_json(json_dict,save)
	if dataset=='NWPU':
		json_dict=NWPU_Dataset(image_path,annotation_path,0)
		save_json(json_dict,save)
	if dataset=='DIOR':
		json_dict=DIOR_Dataset(image_path,annotation_path,0)
		save_json(json_dict,save)
