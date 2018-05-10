import face_recognition as fr
from skimage.transform import warp, SimilarityTransform, resize
import numpy as np

def face_alignment(img, scale=0.9, face_size=(224,224)):
    '''
    face alignment API for single image, get the landmark of eyes and nose and do warpaffine transformation
    :param face_img: single image that including face, I recommend to use dlib frontal face detector
    :param scale: scale factor to judge the output image size

    :return: an aligned single face image
    '''
    h, w, c = img.shape
    output_img = list()
    face_loc_list = fr.face_locations(img)
    for face_loc in face_loc_list:
        face_img = _crop_face(img, face_loc, padding_size=int((face_loc[2] - face_loc[0])*0.5))
        face_land = fr.face_landmarks(face_img)
        if len(face_land) == 0:
            return []
        left_eye_center = _find_center_pt(face_land[0]['left_eye'])
        right_eye_center = _find_center_pt(face_land[0]['right_eye'])
        trotate = _get_rotation_matrix(left_eye_center, right_eye_center, img, scale=scale)
        warped = warp(face_img, trotate)
        warped = (warped*255).astype(np.uint8)
        new_face_loc = fr.face_locations(warped)
        if len(new_face_loc) == 0:
            return []
        output_img.append(resize(_crop_face(warped, new_face_loc[0]), face_size))

    return output_img

def _find_center_pt(points):
    '''
    find centroid point by several points that given
    '''
    x = 0
    y = 0
    num = len(points)
    for pt in points:
        x += pt[0]
        y += pt[1]
    x //= num
    y //= num
    return (x,y)

def _angle_between_2_pt(p1, p2):
    '''
    to calculate the angle rad by two points
    '''
    x1, y1 = p1
    x2, y2 = p2
    tan_angle = (y2 - y1) / (x2 - x1)
    return np.deg2rad(np.degrees(np.arctan(tan_angle)))

def _get_rotation_matrix(left_eye_pt, right_eye_pt, face_img, scale=0.9):
    '''
    to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
    '''
    eye_angle = _angle_between_2_pt(left_eye_pt, right_eye_pt)
    trotate = SimilarityTransform(rotation=eye_angle, scale=scale)
    return trotate

def _dist_nose_tip_center_and_img_center(nose_pt, img_shape):
    '''
    find the distance between nose tip's centroid and the centroid of original image
    '''
    y_img, x_img, _ = img_shape
    img_center = (x_img//2, y_img//2)
    return ((img_center[0] - nose_pt[0]), -(img_center[1] - nose_pt[1]))

def _crop_face(img, face_loc, padding_size=0):
    '''
    crop face into small image, face only, but the size is not the same
    '''
    h, w, c = img.shape
    top = face_loc[0] - padding_size
    right = face_loc[1] + padding_size
    down = face_loc[2] + padding_size
    left = face_loc[3] - padding_size

    if top < 0:
        top = 0
    if right > w - 1:
        right = w - 1
    if down > h - 1:
        down = h - 1
    if left < 0:
        left = 0
    img_crop = img[top:down, left:right]
    return img_crop