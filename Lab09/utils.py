import cv2
import numpy as np

def get_pixel(img, center, x, y):
    if img[x][y] >= center:
        return 1
    else:
        return 0

def ROR(value, nBits, k):
    return ((value<<k)&(2**nBits-1))|(value>>(nBits-k))


def lbp_algorithm(img, x, y, rotation_invariance = False):
    center = img[x][y]
    val_ar = []

    # 주변 픽셀 값 비교
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))      # top
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))      # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))      # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))      # left

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    if rotation_invariance:
        final_value = val
        for k in range(8):
            final_value = min(final_value, ROR(val, 8, k))
        val = final_value

    return val


def img2histo(img):
  height = len(img[0])
  width = len(img[0])
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_lbp = np.zeros((width, height), np.uint8)

  for i in range(1, height-1):
      for j in range(1, width-1):
          ## 채우기
          img_lbp[i, j] = lbp_algorithm(img_gray, i, j, True)  # LBP 알고리즘 적용

  img_histo = cv2.calcHist([img_lbp], [0], None, [256], [0,256])

  return img_histo.flatten()


def img2histo_patch(img, patch_num=5):

  total_histo = np.array([])
  height = len(img[0])//patch_num
  width = len(img[0])//patch_num

  for x in range(patch_num):
    for y in range(patch_num):
      ## part 이미지 선언 후 gray scale로 변경
      part_img = img[x * width:(x + 1) * width, 
                                y * height:(y + 1) * height]
      part_img = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)

      part_lbp = np.zeros((width, height), np.uint8)

      for i in range(1, height-1):
          for j in range(1, width-1):
              part_lbp[i, j] = lbp_algorithm(part_img, i, j, False)

      part_histo = cv2.calcHist([part_lbp], [0], None, [256], [0,256])
      total_histo=np.append(total_histo, part_histo.flatten(), axis = 0)

  return total_histo