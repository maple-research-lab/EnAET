from PIL import Image, ImageEnhance
from PIL import __version__ as PILLOW_VERSION
import PIL
import numpy as np
import os
import math
import random
from torchvision.transforms.functional import _get_inverse_affine_matrix

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def get_params(degrees, translate, scale_ranges, shears, img_size):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    # random generate angle,translate, scale and shear in a range
    angle = random.uniform(degrees[0], degrees[1])
    if translate is not None:
        max_dx = translate[0] * img_size[0]
        max_dy = translate[1] * img_size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = random.uniform(scale_ranges[0], scale_ranges[1])
    else:
        scale = 1.0

    if shears is not None:
        shear = random.uniform(shears[0], shears[1])
    else:
        shear = 0.0

    return angle, translations, scale, shear
image = Image.open('test.jpg')
root_path=os.path.join(os.getcwd(),'Spatial_Trans_Img')
if not os.path.exists(root_path):
    os.mkdir(root_path)
img1_transform=image
width, height = img1_transform.size
center = (img1_transform.size[0] * 0.5 + 0.5, img1_transform.size[1] * 0.5 + 0.5)
shift=4
scale=(0.8,1.2)
shift = [float(random.randint(-int(shift), int(shift))) for ii in range(8)]
scale = random.uniform(scale[0], scale[1])
rotation = random.randint(0, 3)

pts = [((0 - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (0 - center[1]) * scale + center[1]),
               ((width - center[0]) * scale + center[0], (height - center[1]) * scale + center[1]),
               ((0 - center[0]) * scale + center[0], (height - center[1]) * scale + center[1])]
pts = [pts[(ii + rotation) % 4] for ii in range(4)]
pts = [(pts[ii][0] + shift[2 * ii], pts[ii][1] + shift[2 * ii + 1]) for ii in range(4)]

coeffs = find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )
fillcolor=(128, 128, 128)
listIm=[]
listIm.append('test.jpg')
kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
resample=PIL.Image.BILINEAR
img2 = img1_transform.transform((width, height), Image.PERSPECTIVE, coeffs, resample, **kwargs)
tmp_path=os.path.join(root_path,'projective.png')
img2.save(tmp_path)
listIm.append(tmp_path)
degrees=(-180,180)
translate=(-0.2,0.2)
scale=(0.8,1.2)
shear=(-30,30)
ret = get_params(degrees, translate, scale, shear, img1_transform.size)
output_size = img1_transform.size
center = (img1_transform.size[0] * 0.5 + 0.5, img1_transform.size[1] * 0.5 + 0.5)
matrix = _get_inverse_affine_matrix(center, *ret)
kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
img3 = img1_transform.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)
tmp_path=os.path.join(root_path,'affine.png')
img3.save(tmp_path)
listIm.append(tmp_path)
matrix = _get_inverse_affine_matrix(center, ret[0],ret[1],ret[2],0)
kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
img4 = img1_transform.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)
tmp_path=os.path.join(root_path,'similarity.png')
img4.save(tmp_path)
listIm.append(tmp_path)

matrix = _get_inverse_affine_matrix(center, ret[0],ret[1],1.0,0)
kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] == '5' else {}
img4 = img1_transform.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)
tmp_path=os.path.join(root_path,'euclidean.png')
img4.save(tmp_path)
listIm.append(tmp_path)
column = 5
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
target.save('combine-spatial.jpg')
