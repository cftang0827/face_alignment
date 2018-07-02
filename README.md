# face_alignment
Simple face alignment library by using face_recognition and opencv

## Environment
1. MacOS
2. Ubuntu 16.04

## Prerequisite
1. face_recognition 
2. opencv

if there's any problem in environment setting, please check my another repository

https://github.com/cftang0827/python-computer-vision-env_install

## Workflow
1. Use face_recognition(dlib) to find 68 face landmark points.
2. Find the middle point of left eye and right eye
3. Use opencv's warpaffine to correct original image

```
img2_r = io.imread('test2.jpg')
t1 = timeit.default_timer()
img2_a = api.face_alignment(img2_r, scale = 1.05)
print('Time elapsed: {}'.format(timeit.default_timer() - t1))
```

#### Origin image
![Alt](https://github.com/cftang0827/face_alignment/blob/master/original_img.jpg)

#### After alignment image
![Alt](https://github.com/cftang0827/face_alignment/blob/master/after_align.jpg)

```
Time elapsed: 0.200452
```


