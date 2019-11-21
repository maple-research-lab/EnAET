from PIL import Image, ImageEnhance
import numpy as np
import os
import math

"""Sets level to be between 0.2 and 1.8 for ImageEnhance transforms of PIL."""
image = Image.open('test.jpg')
root_path=os.path.join(os.getcwd(),'CCBS_Trans_Img')
if not os.path.exists(root_path):
    os.mkdir(root_path)
listIm=[]
listIm.append('test.jpg')
scale_value1=0.5
scale_value2 = 1.5
scale_value3 = 0.5
scale_value4 = 1.8
image1 = ImageEnhance.Color(image).enhance(scale_value1)
tmp_path=os.path.join(root_path,'Color'+str(scale_value1)+'.png')
listIm.append(tmp_path)
image1.save(tmp_path)
image1 = ImageEnhance.Contrast(image).enhance(scale_value2)
tmp_path=os.path.join(root_path,'Contrast'+str(scale_value2)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
image1 = ImageEnhance.Brightness(image1).enhance(scale_value3)
tmp_path=os.path.join(root_path,'Bright'+str(scale_value3)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
image1 = ImageEnhance.Sharpness(image).enhance(scale_value4)
tmp_path=os.path.join(root_path,'Sharp'+str(scale_value4)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
image1 = ImageEnhance.Contrast(image1).enhance(scale_value2)
tmp_path=os.path.join(root_path,'Color'+str(scale_value1)+'Contrast'+str(scale_value2)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
image1 = ImageEnhance.Brightness(image1).enhance(scale_value3)
tmp_path=os.path.join(root_path,'Color'+str(scale_value1)+'Contrast'+str(scale_value2)+'Bright'+str(scale_value3)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
image1 = ImageEnhance.Sharpness(image1).enhance(scale_value4)
tmp_path=os.path.join(root_path,'Color'+str(scale_value1)+'Contrast'+str(scale_value2)+'Bright'+str(scale_value3)+'Sharp'+str(scale_value4)+'.png')
image1.save(tmp_path)
listIm.append(tmp_path)
column = 4
width = 530
height = 663
size = (530, 663)
imgs = [Image.open(i) for i in listIm]

row_num = math.ceil(len(imgs)/column)
target = Image.new('RGB', (width*column, height*row_num))
for i in range(len(listIm)):
    if i % column == 0:
        end = len(listIm) if i + column > len(listIm) else i + column
        for col, image in enumerate(imgs[i:i+column]):
            target.paste(image, (width*col, height*(i//column),
                                 width*(col + 1), height*(i//column + 1)))
#target.show()
target.save('combine.jpg')
