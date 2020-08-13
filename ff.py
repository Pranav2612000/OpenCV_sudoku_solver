import PIL
import time
import cv2
import operator
import pytesseract
import tensorflow as tf
import math
import tkinter
import pyocr
import pyocr.builders
import cv2
import numpy as np
from scipy import ndimage
from tkinter import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageTk
from tkinter import filedialog
from matplotlib import pyplot as plt
import joblib
import os
from sklearn.svm import LinearSVC
grid_size = 81

def isFull (grid):
    return grid.count('') == 0
  
# can be used more purposefully
def getTrialCelli(grid):
    for i in range(grid_size):
        if grid[i] == '':
            print('trial cell', i)
            return i
      
    
def isLegal(trialVal, trialCelli, grid):
    cols = 0
    for eachSq in range(9):
        trialSq = [ x+cols for x in range(3) ] + [ x+9+cols for x in range(3) ] + [ x+18+cols for x in range(3) ]
        cols +=3
        if cols in [9, 36]:
            cols +=18
        if trialCelli in trialSq:
            for i in trialSq:
                if grid[i] != '':
                    if trialVal == int(grid[i]):
                        print('SQU',end='')
                        return False
  
    for eachRow in range(9):
        trialRow = [ x+(9*eachRow) for x in range (9) ]
        if trialCelli in trialRow:
            for i in trialRow:
                if grid[i] != '':
                    if trialVal == int(grid[i]):
                        print('ROW',end='')
                        return False
  
    for eachCol in range(9):
        trialCol = [ (9*x)+eachCol for x in range (9) ]
        if trialCelli in trialCol:
            for i in trialCol:
                if grid[i] != '':
                    if trialVal == int(grid[i]):
                        print('COL',end='')
                        return False
    print ('is legal', 'cell',trialCelli, 'set to ', trialVal)
    return True

def setCell(trialVal, trialCelli, grid):
    grid[trialCelli] = trialVal
    return grid

def clearCell( trialCelli, grid ):
    grid[trialCelli] = ''
    print('clear cell', trialCelli)
    return grid


def hasSolution (grid):
    if isFull(grid):
        print ('\nSOLVED')
        return True
    else:
        trialCelli = getTrialCelli(grid)
        trialVal = 1
        solution_found = False
        while ( solution_found != True) and (trialVal < 10):
            print ('trial valu',trialVal, end='')
            if isLegal(trialVal, trialCelli, grid):
                grid = setCell(trialVal, trialCelli, grid)
                if hasSolution (grid) == True:
                    solution_found = True
                    return True
                else:
                    clearCell( trialCelli, grid )
            print('++')
            trialVal += 1
    return solution_found


"""
digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype=np.float32)

k = np.arange(10)
cells_labels = np.repeat(k, 250)


test_digits = np.vsplit(test_digits, 50)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32)


# KNN
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=5)


print(result)
"""

#import tesseract
"""
#Load the training image 
img = cv2.imread("digits.png")
#Convert this Image in gray scale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train data and test data.
train = x[:,:50].reshape(-1,400).astype(np.float32)   # Size = (2500,400)

test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]

test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=5
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)

ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print (accuracy)

np.savez('knn_data.npz',train=train, train_labels=train_labels)
"""


"""
# Generate training set
TRAIN_PATH = "Train"
list_folder = os.listdir(TRAIN_PATH)
trainset = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TRAIN_PATH, folder))
    for f in flist:
        im = cv2.imread(os.path.join(TRAIN_PATH, folder, f))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY )
        im = cv2.resize(im, (36,36))
        trainset.append(im)
# Labeling for trainset
train_label = []
for i in range(0,10):
    temp = 500*[i]
    train_label += temp

# Generate testing set
TEST_PATH = "Test"
list_folder = os.listdir(TEST_PATH)
testset = []
test_label = []
for folder in list_folder:
    flist = os.listdir(os.path.join(TEST_PATH, folder))
    for f in flist:
        im = cv2.imread(os.path.join(TEST_PATH, folder, f))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY )
        im = cv2.resize(im, (36,36))
        testset.append(im)
        test_label.append(int(folder))
trainset = np.reshape(trainset, (5000, -1))

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(trainset, train_label)
print("Training finished successfully")

# Testing
testset = np.reshape(testset, (len(testset), -1))
y = clf.predict(testset)
print("Testing accuracy: " + str(clf.score(testset, test_label)))

joblib.dump(clf, "classifier.pkl", compress=3)
"""




def crop_and_warp(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
	distance_between(bottom_right, top_right),
	distance_between(top_left, bottom_left),
	distance_between(bottom_right, bottom_left),
	distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image tofit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def find_corners_of_largest_polygon(img):
    _, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc

def cut_from_rect(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    h, w = img.shape[:2]
    def centre_pad(length):
        if length % 2 == 0:
            """hi"""
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)
    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

	    # Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square
    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)
    #print(pytesseract.image_to_string(digit,config = config))
    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)

def get_digits(img, squares, size):
    digits = []
    sudoko = []
    config = ('--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789 -l eng tsv')
    img = pre_process_image(img.copy(), skip_dilate=True)
    dim["text"] = "hello"
    for square in squares:
        digit = extract_digit(img, square, size)
        #plt.imshow(digit)
        #plt.show()
        #digit = cv2.resize(digit, (10,10))
        #digit = cv2.bitwise_not(digit, digit)
        #dig = Image.fromarray(digit)
        #sudoko.append(pytesseract.image_to_string(dig,config = config))
        digits.append(digit)
    #dim["text"] = sudoko
    #print (sudoko)
    return digits

def infer_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    for i in range(9):
        for j in range(9):
            p1 = (j * side, i * side)
            p2 = ((j + 1) * side, (i + 1) * side)
            squares.append((p1, p2))
    return squares
def distance_between(pt1, pt2):
    a = pt1[0] - pt2[0]
    b = pt1[1] - pt2[1]
    return np.sqrt((a ** 2) + (b ** 2))
def calc_scale(height, width):
    h_sc = int(500 / height)
    w_sc = int(250 / width)
    return min(h_sc, w_sc)
def get_confidence(prediction):
    j = len(prediction) - 1
    confidence = 0
    while(prediction[j] != '\t'):
        j = j - 1
    j = j - 1
    while(prediction[j] != '\t'):
        j = j - 1
    j = j + 1
    while(prediction[j] != '\t'):
        confidence = confidence * 10 + int(prediction[j])
        j = j + 1
    return confidence

def get_digit(prediction):
    j = len(prediction) - 1
    digit = ""
    while(prediction[j] != '\t'):
        j = j - 1
    j = j + 1
    while j < len(prediction) and prediction[j] != '\t':
        digit = digit + prediction[j]
        j = j + 1
    return digit

def get_predicted_digit(prediction, dig_img):
    """
    if rotation by 90 char contains -(90%) or L :
        identify as 1
    if rotation by 90 char is =(90%) or J or has < has +:
        identify as 4
    if rotation by 180 char is J or t or 7:#L UNSURE
        identify as 4
    if rotation by 180 char is i
        identify as 1
    if rotation by 270 is -:(100%)
        identify as 1
    if rotation by 270 is >:
        identify as 4
    if vflip char is t:
        identify as 1
    if hflip char has J or V:#UNSURE
        identify as 4
    if hflip char has t:
        identify as 4
    """
    print(prediction)
    confidence = get_confidence(prediction)
    digit = get_digit(prediction)
    config = ('--oem 1 --psm 10 -l eng')
    if len(digit) == 1:
        if digit.isdigit():
            if digit == '4':
                print(confidence)
                chance4 = 0
                if confidence < 45:
                    return 7
                if confidence > 92:
                    chance4 = chance4 + 4
                if confidence > 85:
                    chance4 = chance4 + 2
                if confidence < 80:
                    chance4 = chance4 - 2
                (height, width) = dig_img.shape[:2]
                center = (width/2, height/2)
                dig_img_orig = dig_img.copy()

                #Rotate by 90
                R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
                predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
                digit_rot_90 = get_digit(predicted_rot_90) 
                print(digit_rot_90)
                not3 = False
                if digit_rot_90.find('-') != -1:
                    print("- identified")
                    not3 = True
                    chance4 = chance4 - 4 # 3 to be replaced by other value
                if digit_rot_90.find('—') != -1:
                    print("- identified")
                    not3 = True
                    chance4 = chance4 - 4 # 3 to be replaced by other value
                if digit_rot_90 == 'L':
                    print("L identified")
                    chance4 = chance4 - 1
                if digit_rot_90.find('=') != -1 or digit_rot_90.find('+') != -1 or digit_rot_90.find('<') != -1:
                    print("< identified")
                    not3 = True
                    chance4 = chance4 + 3 # 3 to be replaced by other value
                if digit_rot_90.find('J') != -1:
                    print("J identified")
                    chance4 = chance4 + 1

                #Rotate by 180
                R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
                predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
                digit_rot_180 = get_digit(predicted_rot_180) 
                print(digit_rot_180)
                if digit_rot_180.find('t') != -1:
                    print("t identified 180")
                    chance4 = chance4 + 1
                if digit_rot_180.find('7') != -1:
                    print("7 identified")
                    chance4 = chance4 + 1
                if digit_rot_180 == 'i':
                    print("i identified")
                    chance4 = chance4 - 2
                if digit_rot_180.find('b') != -1:
                    chance4 = chance4 + 3
                if digit_rot_180.find('E') != -1 and not3 == False and len(digit_rot_180) < 3:
                    return 3

                #Rotate by 270 or -90
                R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
                rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
                predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
                digit_rot_270 = get_digit(predicted_rot_270) 
                print(digit_rot_270)
                if digit_rot_270.find('-') != -1:
                    print("- identified")
                    chance4 = chance4 - 4
                    if digit_rot_90.find('-') != -1:
                        return 1;
                    if digit_rot_90.find('—') != -1:
                        return 1;
                if digit_rot_270.find('—') != -1:
                    print("- identified")
                    chance4 = chance4 - 4
                    if digit_rot_90.find('-') != -1:
                        return 1;
                    if digit_rot_90.find('—') != -1:
                        return 1;
                if digit_rot_270.find('>') != -1:
                    print("> identified")
                    chance4 = chance4 + 1
                    
                #Vertical Mirror
                vflip = cv2.flip(dig_img,1)
                predicted_vflip = pytesseract.image_to_data(vflip, config = config)
                digit_vflip = get_digit(predicted_vflip)
                print(digit_vflip)
                if digit_vflip.find('t') != -1:
                    print("t identified -1")
                    chance4 = chance4 - 2
                if digit_vflip.find('E') != -1 and not3 == False and len(digit_vflip) < 3:
                    return 3
                #Horizontal Mirror
                hflip = cv2.flip(dig_img,0)
                predicted_hflip = pytesseract.image_to_data(hflip, config = config)
                digit_hflip = get_digit(predicted_hflip)
                print(digit_hflip)
                if digit_hflip.find('t') != -1:
                    print("t identified")
                    chance4 = chance4 + 1
                if digit_hflip.find('J') != -1:
                    print("J identified")
                    chance4 = chance4 + 1
                if digit_hflip.find('V') != -1:
                    print("V identified")
                    chance4 = chance4 + 3

                print("Chance is: " + str(chance4))
                if chance4 > 0:
                    return 4
                else:
                    return 1


            else:
                return digit
                #Code to get differentiate 1, 4, 7
        else:
            if digit == 'Q':
                return 9
            if digit == 'g':
                print(confidence)
                chance9 = 0
                if confidence > 92:
                    chance9 = chance9 + 4
                if confidence > 85:
                    chance9 = chance9 + 2
                if confidence < 80:
                    chance9 = chance9 - 2
                (height, width) = dig_img.shape[:2]
                center = (width/2, height/2)
                dig_img_orig = dig_img.copy()

                #Rotate by 90
                R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
                predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
                digit_rot_90 = get_digit(predicted_rot_90) 
                print(digit_rot_90)

                
                #Rotate by 180
                R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
                predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
                digit_rot_180 = get_digit(predicted_rot_180) 
                print(digit_rot_180)
                if digit_rot_180[0] == '6':
                    chance9 = chance9 + 4
                if digit_rot_180[0] == 'B':
                    chance9 = chance9 - 4
                if digit_rot_180.find('E') != -1:
                    return 3

                #Rotate by 270 or -90
                R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
                rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
                predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
                digit_rot_270 = get_digit(predicted_rot_270) 
                print(digit_rot_270)

                if digit_rot_270 == "oo" or digit_rot_270 == "co":
                        chance9 = chance9 - 3
                    
                #Vertical Mirror
                vflip = cv2.flip(dig_img,1)
                predicted_vflip = pytesseract.image_to_data(vflip, config = config)
                digit_vflip = get_digit(predicted_vflip)
                print(digit_vflip)
                if digit_vflip.find('E') != -1:
                    return 3
                if digit_vflip.find('e') != -1:
                    chance9 = chance9 + 3
                if digit_vflip.find('B') != -1:
                    chance9 = chance9 - 4
                if digit_vflip.find('8') != -1:
                    chance9 = chance9 - 4
                #Horizontal Mirror
                hflip = cv2.flip(dig_img,0)
                predicted_hflip = pytesseract.image_to_data(hflip, config = config)
                digit_hflip = get_digit(predicted_hflip)
                print(digit_hflip)
                if digit_hflip.find('a') != -1:
                    chance9 = chance9 + 3
                if digit_hflip.find('g') != -1:
                    chance9 = chance9 - 4
                if digit_hflip.find('B') != -1:
                    chance9 = chance9 - 4
                if digit_hflip.find('8') != -1:
                    chance9 = chance9 - 4

                print("Chance is: " + str(chance9))
                if chance9 > 0:
                    return 9
                else:
                    return 8
            if digit == 'q':
                print(confidence)
                chance4 = 0
                (height, width) = dig_img.shape[:2]
                center = (width/2, height/2)
                dig_img_orig = dig_img.copy()

                #Rotate by 90
                R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
                predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
                digit_rot_90 = get_digit(predicted_rot_90) 
                print(digit_rot_90)
                if digit_rot_90.find('=') != -1 or digit_rot_90.find('+') != -1 or digit_rot_90.find('<') != -1:
                    print("< identified")
                    chance4 = chance4 + 3 # 3 to be replaced by other value
                if digit_rot_90.find('J') != -1:
                    print("J identified")
                    chance4 = chance4 + 1

                #Rotate by 180
                R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
                predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
                digit_rot_180 = get_digit(predicted_rot_180) 
                print(digit_rot_180)
                if digit_rot_180.find('t') != -1:
                    print("t identified 180")
                    chance4 = chance4 + 1
                if digit_rot_180.find('7') != -1:
                    print("7 identified")
                    chance4 = chance4 + 1
                if digit_rot_180.find('6') != -1:
                    chance4 = chance4 - 4

                #Rotate by 270 or -90
                R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
                rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
                predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
                digit_rot_270 = get_digit(predicted_rot_270) 
                print(digit_rot_270)
                if digit_rot_270.find('>') != -1:
                    print("> identified")
                    chance4 = chance4 + 1
                    
                #Vertical Mirror
                vflip = cv2.flip(dig_img,1)
                predicted_vflip = pytesseract.image_to_data(vflip, config = config)
                digit_vflip = get_digit(predicted_vflip)
                print(digit_vflip)
                if digit_vflip.find('e') != -1:
                    chance4 = chance4 - 3
                #Horizontal Mirror
                hflip = cv2.flip(dig_img,0)
                predicted_hflip = pytesseract.image_to_data(hflip, config = config)
                digit_hflip = get_digit(predicted_hflip)
                print(digit_hflip)
                if digit_hflip.find('t') != -1:
                    print("t identified")
                    chance4 = chance4 + 1
                if digit_hflip.find('J') != -1:
                    print("J identified")
                    chance4 = chance4 + 1
                if digit_hflip.find('V') != -1:
                    print("V identified")
                    chance4 = chance4 + 3
                if digit_hflip.find('a') != -1:
                    chance4 = chance4 - 3

                print("Chance is: " + str(chance4))
                if chance4 >= 0:
                    return 4
                else:
                    return 9
            elif digit == '?':
                return 2
            elif digit == 'a':
                print(confidence)
                chance8 = 0
                (height, width) = dig_img.shape[:2]
                center = (width/2, height/2)
                dig_img_orig = dig_img.copy()

                #Rotate by 90
                R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
                predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
                digit_rot_90 = get_digit(predicted_rot_90) 
                print(digit_rot_90)
                if digit_rot_90.find('y') != -1:
                    chance8 = chance8 - 1
                if digit_rot_90.find('=') != -1 or digit_rot_90.find('+') != -1 or digit_rot_90.find('<') != -1:
                    print("< identified")
                    chance8 = chance8 - 3 # 3 to be replaced by other value
                if digit_rot_90.find('J') != -1:
                    print("J identified")
                    chance8 = chance8 - 1

                #Rotate by 180
                R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
                predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
                digit_rot_180 = get_digit(predicted_rot_180) 
                print(digit_rot_180)
                if digit_rot_180[0] == 'B':
                    chance8 = chance8 + 4
                if digit_rot_180.find('t') != -1:
                    print("t identified 180")
                    chance8 = chance8 - 1
                if digit_rot_180.find('7') != -1:
                    print("7 identified")
                    chance8 = chance8 - 1

                #Rotate by 270 or -90
                R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
                rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
                predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
                digit_rot_270 = get_digit(predicted_rot_270) 
                print(digit_rot_270)
                if digit_rot_270 == 'oo' or digit_rot_270 == "co":
                    chance8 = chance8 + 4
                if digit_rot_270.find('>') != -1:
                    print("> identified")
                    chance8 = chance8 - 1
                    
                #Vertical Mirror
                vflip = cv2.flip(dig_img,1)
                predicted_vflip = pytesseract.image_to_data(vflip, config = config)
                digit_vflip = get_digit(predicted_vflip)
                print(digit_vflip)
                if digit_vflip.find('B') != -1:
                    chance8 = chance8 + 2
                if digit_vflip.find('8') != -1:
                    chance8 = chance8 + 4
                #Horizontal Mirror
                hflip = cv2.flip(dig_img,0)
                predicted_hflip = pytesseract.image_to_data(hflip, config = config)
                digit_hflip = get_digit(predicted_hflip)
                print(digit_hflip)
                if digit_vflip.find('E') != -1:
                    return 3
                if digit_hflip.find('t') != -1:
                    print("t identified")
                    chance8 = chance8 - 1
                if digit_hflip.find('J') != -1:
                    print("J identified")
                    chance8 = chance8 - 1
                if digit_hflip.find('q') != -1:
                    print("J identified")
                    chance8 = chance8 - 1
                if digit_hflip.find('V') != -1:
                    print("V identified")
                    chance8 = chance8 - 3
                print("Chance is: " + str(chance8))
                if chance8 > 0:
                    return 8
                else:
                    return 4
            elif digit == 'B':
                return 8
            elif digit == 'S' or digit == 's' or digit == '§':
                return 5
            elif digit == 't':
                return 7
            elif digit == 'A':
                return 4
            elif digit == 'j' or digit == 'I':
                return 1

            #Code for basic char to digit conversions e.g. 'g' to '9'
    else:
        if digit == '41' or digit == '14' or digit == '4d':
            return 1
        if digit.find('?') != -1:
            return 2
        if digit.find('S') != -1:
            return 5
        if digit.find('j') != -1:
            return 1
        if digit.find('I') != -1:
            return 1
        if digit.find('s') != -1:
            return 5
        if digit.find('t') != -1:
            return 7
        if digit.find('a') != -1:
            print(confidence)
            chance8 = 0
            (height, width) = dig_img.shape[:2]
            center = (width/2, height/2)
            dig_img_orig = dig_img.copy()

            #Rotate by 90
            R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
            rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
            predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
            digit_rot_90 = get_digit(predicted_rot_90) 
            print(digit_rot_90)
            if digit_rot_90.find('y') != -1:
                chance8 = chance8 - 1
            if digit_rot_90.find('=') != -1 or digit_rot_90.find('+') != -1 or digit_rot_90.find('<') != -1:
                print("< identified")
                chance8 = chance8 - 3 # 3 to be replaced by other value
            if digit_rot_90.find('J') != -1:
                print("J identified")
                chance8 = chance8 - 1

            #Rotate by 180
            R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
            rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
            predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
            digit_rot_180 = get_digit(predicted_rot_180) 
            print(digit_rot_180)
            if digit_rot_180[0] == 'B':
                chance8 = chance8 + 4
            if digit_rot_180.find('t') != -1:
                print("t identified 180")
                chance4 = chance4 - 1
            if digit_rot_180.find('7') != -1:
                print("7 identified")
                chance4 = chance4 - 1

            #Rotate by 270 or -90
            R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
            rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
            predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
            digit_rot_270 = get_digit(predicted_rot_270) 
            print(digit_rot_270)
            if digit_rot_270 == 'oo' or digit_rot_270 == "co":
                chance8 = chance8 + 4
            if digit_rot_270.find('>') != -1:
                print("> identified")
                chance8 = chance8 - 1
                
            #Vertical Mirror
            vflip = cv2.flip(dig_img,1)
            predicted_vflip = pytesseract.image_to_data(vflip, config = config)
            digit_vflip = get_digit(predicted_vflip)
            print(digit_vflip)
            if digit_vflip.find('B') != -1:
                chance8 = chance8 + 2
            if digit_vflip.find('8') != -1:
                chance8 = chance8 + 4
            #Horizontal Mirror
            hflip = cv2.flip(dig_img,0)
            predicted_hflip = pytesseract.image_to_data(hflip, config = config)
            digit_hflip = get_digit(predicted_hflip)
            print(digit_hflip)
            if digit_vflip.find('E') != -1:
                return 3
            if digit_hflip.find('t') != -1:
                print("t identified")
                chance8 = chance8 - 1
            if digit_hflip.find('J') != -1:
                print("J identified")
                chance8 = chance8 - 1
            if digit_hflip.find('q') != -1:
                print("J identified")
                chance8 = chance8 - 1
            if digit_hflip.find('V') != -1:
                print("V identified")
                chance8 = chance8 - 3
            print("Chance is: " + str(chance8))
            if chance8 > 0:
                return 8
            else:
                return 4

        if digit == '[>':
            return 2
        for i in range(9):
            if digit.find(str(i + 1)) != -1:
                if i == 3:

                    print(confidence)
                    chance4 = 0
                    if confidence < 45:
                        return 7
                    if confidence > 92:
                        chance4 = chance4 + 4
                    if confidence > 85:
                        chance4 = chance4 + 2
                    if confidence < 80:
                        chance4 = chance4 - 2
                    (height, width) = dig_img.shape[:2]
                    center = (width/2, height/2)
                    dig_img_orig = dig_img.copy()

                    #Rotate by 90
                    R90 = cv2.getRotationMatrix2D(center, 90, 1.0)
                    rotated_90 = cv2.warpAffine(dig_img, R90, (width,height))
                    predicted_rot_90 = pytesseract.image_to_data(rotated_90, config = config)
                    digit_rot_90 = get_digit(predicted_rot_90) 
                    print(digit_rot_90)
                    if digit_rot_90.find('-') != -1:
                        print("- identified")
                        chance4 = chance4 - 4 # 3 to be replaced by other value
                    if digit_rot_90.find('—') != -1:
                        print("- identified")
                        chance4 = chance4 - 4 # 3 to be replaced by other value
                    if digit_rot_90 == 'L':
                        print("L identified")
                        chance4 = chance4 - 1
                    if digit_rot_90.find('=') != -1 or digit_rot_90.find('+') != -1 or digit_rot_90.find('<') != -1:
                        print("< identified")
                        chance4 = chance4 + 3 # 3 to be replaced by other value
                    if digit_rot_90.find('J') != -1:
                        print("J identified")
                        chance4 = chance4 + 1

                    #Rotate by 180
                    R180 = cv2.getRotationMatrix2D(center, 180, 1.0)
                    rotated_180 = cv2.warpAffine(dig_img, R180, (width,height))
                    predicted_rot_180 = pytesseract.image_to_data(rotated_180, config = config)
                    digit_rot_180 = get_digit(predicted_rot_180) 
                    print(digit_rot_180)
                    if digit_rot_180.find('t') != -1:
                        print("t identified 180")
                        chance4 = chance4 + 1
                    if digit_rot_180.find('7') != -1:
                        print("7 identified")
                        chance4 = chance4 + 1
                    if digit_rot_180 == 'i':
                        print("i identified")
                        chance4 = chance4 - 2
                    if digit_rot_180.find('E') != -1:
                        return 3

                    #Rotate by 270 or -90
                    R270 = cv2.getRotationMatrix2D(center, -90, 1.0)
                    rotated_270 = cv2.warpAffine(dig_img, R270, (width,height))
                    predicted_rot_270 = pytesseract.image_to_data(rotated_270, config = config)
                    digit_rot_270 = get_digit(predicted_rot_270) 
                    print(digit_rot_270)
                    if digit_rot_270.find('-') != -1:
                        print("- identified")
                        chance4 = chance4 - 4
                        if digit_rot_90.find('-') != -1:
                            return 1;
                        if digit_rot_90.find('—') != -1:
                            return 1;
                    if digit_rot_270.find('—') != -1:
                        print("- identified")
                        chance4 = chance4 - 4
                        if digit_rot_90.find('-') != -1:
                            return 1;
                        if digit_rot_90.find('—') != -1:
                            return 1;
                    if digit_rot_270.find('>') != -1:
                        print("> identified")
                        chance4 = chance4 + 1
                        
                    #Vertical Mirror
                    vflip = cv2.flip(dig_img,1)
                    predicted_vflip = pytesseract.image_to_data(vflip, config = config)
                    digit_vflip = get_digit(predicted_vflip)
                    print(digit_vflip)
                    if digit_vflip.find('t') != -1:
                        print("t identified -1")
                        chance4 = chance4 - 2
                    if digit_vflip.find('E') != -1:
                        return 3
                    #Horizontal Mirror
                    hflip = cv2.flip(dig_img,0)
                    predicted_hflip = pytesseract.image_to_data(hflip, config = config)
                    digit_hflip = get_digit(predicted_hflip)
                    print(digit_hflip)
                    if digit_hflip.find('t') != -1:
                        print("t identified")
                        chance4 = chance4 + 1
                    if digit_hflip.find('J') != -1:
                        print("J identified")
                        chance4 = chance4 + 1
                    if digit_hflip.find('V') != -1:
                        print("V identified")
                        chance4 = chance4 + 3

                    print("Chance is: " + str(chance4))
                    if chance4 > 0:
                        return 4
                    else:
                        return 1
                else:
                    return i + 1
    return 0
def printGrid (grid, add_zeros):
    i = 0
    for val in grid:
        if add_zeros == 1:
            if int(val) < 10: 
                print('0'+str(val),end='')
            else:
                print (val,end='')
        else:
            print (val,end='')
        i +=1
        if i in [ (x*9)+3 for x in range(81)] +[ (x*9)+6 for x in range(81)] +[ (x*9)+9 for x in range(81)] :
            print ('|',end='')
        if add_zeros == 1:
            if i in [ 27, 54, 81]:
                print('\n---------+----------+----------+')
            elif i in [ (x*9) for x in range(81)]:
                print('\n')
        else:
            if i in [ 27, 54, 81]:
                print('\n------+-------+-------+')
            elif i in [ (x*9) for x in range(81)]:
                print('\n')

def print_sudoko(sudoko):
    print(" — — — — — — — — — ")
    for i in range(9):
        k = 9 * i
        print("|", end = "")
        for j in range(9):
            if j != 8:
                print(sudoko[k + j], end = "|")
            else:
                print(sudoko[k + j], end = "|\n")
        #print("\n")
    print(" — — — — — — — — — ")
    #print("—  —  —  —  —  —  —  —  —")
    #print("— — — — — — — —")

def select_img():
    global panelA, panelB, dimensions, sudoko
    sudoko = []


    path = filedialog.askopenfilename(initialdir = "/home/Desktop", title = "Select file", filetypes = (("jpeg files", "*.jpeg"), ("all files", "*.*")))
    if len(path) > 0:
        image = cv2.imread(path)
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        dim["text"] = original.shape#"Image Selected"
        if(original.shape[0] > 500 or original.shape[1] > 450):
            #scale_percent = calc_scale(image.shape[0], image.shape[1])
            scale_percent = 0.5
            width = int(original.shape[1] * scale_percent)
            height = int(original.shape[0] * scale_percent)
            new_dimensions = (width, height)
            original = cv2.resize(original, new_dimensions, interpolation = cv2.INTER_AREA)
            image = cv2.resize(image, new_dimensions, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pnlA = Image.fromarray(image)
        resp = display_panelA(img_pnlA)
        if resp == 1:
            print("Sudoko imported")
        processed = pre_process_image(original)
        corners = find_corners_of_largest_polygon(processed)
        cropped = crop_and_warp(original, corners)
        squares = infer_grid(cropped)
        digits = get_digits(cropped, squares, 28)
        """image = cv2.medianBlur(image, 5)
        dim["text"] = image.shape#"Image Selected"
        if(image.shape[0] > 500 or image.shape[1] > 450):
            #scale_percent = calc_scale(image.shape[0], image.shape[1])
            scale_percent = 0.5
            width = int(image.shape[1] * scale_percent)
            height = int(image.shape[0] * scale_percent)
            new_dimensions = (width, height)
            image = cv2.resize(image, new_dimensions, interpolation = cv2.INTER_AREA)
            image1 = cv2.resize(image, new_dimensions, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (9,9), 0)
        #Thresholding
        edged = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edged = cv2.bitwise_not(edged,edged)
        #iedged = cv2.Canny(gray, 50, 100)

        #Dilation
        notted = edged
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        """ """kernel1 = np.ones((5,5), np.uint8)
        notted = cv2.morphologyEx(notted, cv2.MORPH_OPEN, kernel1)"""
        """final = cv2.dilate(notted,kernel)"""
        #Rotation
        """
        img_edges = cv2.Canny(gray, 100, 100, apertureSize = 3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength = 100, maxLineGap = 5)
        angles = []
        #if not(np.array(lines).size):
        if True:
            for x1, y1, x2, y2 in lines[0]:
                cv2.line(final.copy(), (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            median_angle = np.median(angles)
        else:
            median_angle = 0
        rotated = ndimage.rotate(final, median_angle)
"""
        """rotated = final
        #Rotation complete
        #Contours
        new_img, ext_contours, hier = cv2.findContours(rotated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_img, contours, hier = cv2.findContours(rotated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key = operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key = operator.itemgetter(1))


        contour = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)
        all_contours = cv2.drawContours(contour.copy(), contours, -1, (255,0,0), 2) 
        external_only = cv2.drawContours(contour.copy(), ext_contours, -1, (255,0,0), 2) 
        
        src = np.array([[polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]], np.float32)
        side = max([
            distance_between(polygon[bottom_right][0], polygon[top_right][0]),
            distance_between(polygon[top_left][0], polygon[bottom_left][0]),
            distance_between(polygon[bottom_right][0], polygon[bottom_left][0]),
            distance_between(polygon[top_left][0], polygon[top_right][0])
        ])
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], np.float32)
        #src = np.array([[50,50],[450,450],[70,420],[420,70]],np.float32)
        #dst = np.array([[0,0],[299,299],[0,299],[299,0]],np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        cropped = cv2.warpPerspective(image1, m, (int(side), int(side)))
# Gridding
        rects = infer_grid(cropped)
        for rect in rects:
            squares = cv2.rectangle(cropped, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), (0,0,255))

        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cropped = cv2.bitwise_not(cropped, cropped)
        digits = get_digits(cropped, squares, 28)
"""
        rows = []
        with_border = [cv2.copyMakeBorder(img.copy(), 0, 0, 0, 0, cv2.BORDER_CONSTANT, None, 255) for img in digits]
        """
        for i in range(81):
            di = with_border[i]
            config = ('--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789')
            di = cv2.bitwise_not(di, di)
            sudoko =[]
            sudoko.append(pytesseract.image_to_string(di,config = config))
        print(sudoko)
        """
        for i in range(9):
            row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis = 1)
            rows.append(row)
        ans = np.concatenate(rows)
        ans = cv2.bitwise_not(ans, ans)
        di = cv2.bitwise_not(digits[1])
        config = ('--oem 1 --psm 10 -l eng')
        char = []
        sudoko = []
        for di in digits:
            di = cv2.bitwise_not(di)
            prediction = pytesseract.image_to_data(di,config = config)
            #print(i)
            if prediction[len(prediction) - 2] != '-':
                #confidence = get_confidence(prediction)
                dig = get_predicted_digit(prediction, di)
                #print(dig)
                sudoko.append(dig)
            else:
                char.append("-")
                sudoko.append('0')
                #print("blank")

        sudoko_inp = []
        no_of_zeroes = 0
        solved_sud = create_sudoko_image(sudoko, sudoko)
        solved_sud = Image.fromarray(solved_sud)
        display_panelB(solved_sud)
        print_sudoko(sudoko)
        """
        change = input()
        while(change != '-1'):
            print(change)
            X, Y, num = change.split(",")
            sudoko[9 * (int(Y) - 1) + int(X) - 1] = num
            solved_sud = create_sudoko_image(sudoko, sudoko)
            solved_sud = Image.fromarray(solved_sud)
            display_panelB(solved_sud)
            print_sudoko(sudoko)
            change = input() 
            """
        #sampleGrid = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '4', '6', '2', '9', '5', '1', '8', '1', '9', '6', '3', '5', '8', '2', '7', '4', '4', '7', '3', '8', '9', '2', '6', '5', '1', '6', '8', '.', '.', '3', '1', '.', '4', '.', '.', '.', '.', '.', '.', '.', '3', '8', '.']
        """
        for digit in sudoko:
            if digit == 0 or digit == '0':
                no_of_zeroes = no_of_zeroes + 1
                sudoko_inp.append('.')
            else:
                sudoko_inp.append(str(digit))
        sudoko = sudoko_inp.copy()
        if no_of_zeroes < 70 and hasSolution (sudoko_inp):
            print_sudoko(sudoko_inp)
        else:
            print('NO SOLUTION')
        #dim["text"] = prediction
        solved_sud = create_sudoko_image(sudoko, sudoko_inp)
        solved_sud = Image.fromarray(solved_sud)
        display_panelB(solved_sud)
        """
def display_panelA(image):
    global panelA
    image = ImageTk.PhotoImage(image)
    if panelA is None:
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx = 10, pady=10)
    else:
        panelA.configure(image=image)
        panelA.image = image
    return 1
def display_panelB(image):
    global panelB
    image = ImageTk.PhotoImage(image)
    if panelB is None:
        panelB = Label(image=image)
        panelB.image = image 
        panelB.pack(side="right", padx=10, pady=10)
    else:
        panelB.configure(image=image)
        panelB.image = image 
def create_sudoko_image(sudoko, sudoko_inp):
    ans_digits = []
    fnt = ImageFont.truetype("arial.ttf", 45)
    for i in range(len(sudoko_inp)):
        tryimg = Image.new('RGB', (50, 50), color = 'black')
        d = ImageDraw.Draw(tryimg)
        if sudoko[i] == '0' or sudoko[i] == '':
            sudoko[i] = ''
            num_clr = "white"
        else:
            num_clr = "red"
        d.text((0,0), str(sudoko_inp[i]), fill = num_clr, font = fnt)
        tryimg_cv = cv2.cvtColor(np.array(tryimg), cv2.COLOR_RGB2BGR)
        ans_digits.append(tryimg_cv)
    ans_with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 255) for img in ans_digits]
    rows = []
    for i in range(9):
        row = np.concatenate(ans_with_border[i * 9:((i + 1) * 9)], axis = 1)
        rows.append(row)
    solved_sud = np.concatenate(rows)
    return solved_sud
def solve_sudoko():
    print("solving Sudoko....")
    global sudoko, sudoko_inp
    no_of_zeroes = 0
    for digit in sudoko:
        if digit == 0 or digit == '':
            no_of_zeroes = no_of_zeroes + 1
            sudoko_inp.append('')
        else:
            sudoko_inp.append(str(digit))
    sudoko = sudoko_inp.copy()
    if no_of_zeroes < 70 and hasSolution (sudoko_inp):
        print_sudoko(sudoko_inp)
    else:
        print('NO SOLUTION')
    #dim["text"] = prediction
    solved_sud = create_sudoko_image(sudoko, sudoko_inp)
    solved_sud = Image.fromarray(solved_sud)
    display_panelB(solved_sud)
def update_sudoko():
    print("Updating....")
    X = int(Entry.get(x))
    Y = int(Entry.get(y))
    num = int(Entry.get(Num))
    sudoko[9 * (int(Y) - 1) + int(X) - 1] = num
    solved_sud = create_sudoko_image(sudoko, sudoko)
    solved_sud = Image.fromarray(solved_sud)
    display_panelB(solved_sud)
    print_sudoko(sudoko)
    x.delete(0, END)
    y.delete(0, END)
    Num.delete(0, END)
window = tkinter.Tk()
window.geometry("500x650")
#window.resizable(0,0)
window.title("Sudoko")
sudoko = []
sudoko_inp = []
panelA = None
panelB = None

#text = tkinter.Label(window, text = "Hello World!").pack(side = "bottom")
top_frame = tkinter.Frame(window).pack()

dimensions = "No image selected"
bottom_frame = tkinter.Frame(window).pack(side = "bottom")
dim = tkinter.Label(window, text=dimensions)
dim.pack(side="bottom")
cam = tkinter.Button(bottom_frame, text = "Get Image", fg = "white", bg = "black", command=select_img).pack(side = "bottom")
sol = tkinter.Button(bottom_frame, text = "Solve", fg = "white", bg = "black", command=solve_sudoko).pack(side = "bottom")
update = tkinter.Button(bottom_frame, text = "Update", fg = "white", bg = "black", command=update_sudoko).pack(side = "bottom")
x = Entry(window)
y = Entry(window)
Num = Entry(window)
"""
x.grid(row = 0, column = 1)
y.grid(row = 0, column = 1)
num.grid(row = 0, column = 1)
"""
x.pack()
y.pack()
Num.pack()
window.mainloop()
"""

If % less than 40:
    identify as 7:
If % above 92:
    idenitfy as 4:
else:
    if rotation by 90 char contains -(90%) or L :
        identify as 1
    if rotation by 90 char is =(90%) or J or has < has +:
        identify as 4
    if rotation by 180 char is J or t or 7:#L UNSURE
        identify as 4
    if rotation by 180 char is i
        identify as 1
    if rotation by 270 is -:(100%)
        identify as 1
    if rotation by 270 is >:
        identify as 4
    if vflip char is t:
        identify as 1
    if hflip char has J or V:#UNSURE
        identify as 4
    if hflip char has t:
        identify as 4
    else:
    ___________________________________________________________________________________________________________________________________
                confidence = int(i[len(i) - 4]) * 10 + int(i[len(i) - 3])
                if i[len(i) - 1] == 'Q':
                    print("9")
                if i[len(i) - 1] == '?':
                    print("2")
                if i[len(i) - 1] == 'a':
                    print("4")
                else:
                    confidence = int(i[len(i) - 4]) * 10 + int(i[len(i) - 3])
                    print(str(i[len(i) - 1]) + "Confidence:" + str(confidence))
                j = len(i) - 1
                while i[j] != '\t':
                    print(i[j])
                    if(i[j] == '4'):
                        #rotate
                        (h, w) = di.shape[:2]
                        center = (w/2, h/2)
                        orig = di.copy()
                        M = cv2.getRotationMatrix2D(center, 90, 1.0)
                        rotated = cv2.warpAffine(di, M, (w,h))
                        ri = pytesseract.image_to_data(rotated, config = config)
                        print(ri)
                        print("_____________________________")
                        M = cv2.getRotationMatrix2D(center, 180, 1.0)
                        rotated = cv2.warpAffine(di, M, (w,h))
                        ri = pytesseract.image_to_data(rotated, config = config)
                        print(ri)
                        print("_____________________________")
                        M = cv2.getRotationMatrix2D(center, -90, 1.0)
                        rotated = cv2.warpAffine(di, M, (w,h))
                        ri = pytesseract.image_to_data(rotated, config = config)
                        print(ri)
                        print("_____________________________")
                        M = cv2.getRotationMatrix2D(center, -45, 1.0)
                        rotated = cv2.warpAffine(di, M, (w,h))
                        ri = pytesseract.image_to_data(rotated, config = config)
                        print(ri)
                        print("_____________V_______________")
                        vflip = cv2.flip(orig,1)
                        ri = pytesseract.image_to_data(vflip, config = config)
                        print(ri)
                        print("_____________________________")
                        hflip = cv2.flip(di,0)
                        ri = pytesseract.image_to_data(hflip, config = config)
                        print(ri)
                        print("_____________________________")
                    else:
                        char.append(i[j])
                    j = j - 1
                j = j - 1
                conf_str = []
                while i[j] != '\t':
                    conf_str.append(i[j])
                    j = j - 1
                confidence = 0
                conf_str.reverse()
                for ch in conf_str:
                    confidence = confidence * 10 + int(ch)
                print(confidence)
"""
