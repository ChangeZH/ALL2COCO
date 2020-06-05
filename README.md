# ALL2COCO
将所有目标检测数据集标签格式转为COCO标签的json格式。
## 运行参数
   `python 2COCO.py --image_path 所有图片的路径 --annotation_path 所有标签的路径 --dataset 选择数据集 --save json保存路径`

## 目前支持数据集

   NWPU VHR-10(txt)：`--dataset NWPU`
   
   RSOD-Dataset(txt)：`--dataset RSOD`
   
   DIOR-Dataset(xml)：`--dataset DIOR`
   
   YOLO-Dataset(txt)：`--dataset YOLO`
   
## 目前测试成功的项目
   
   Yet-Another-EfficientDet-Pytorch(https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
