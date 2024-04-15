import cv2
import time
import numpy as np

def load_img(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return img

def display_img(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def moravec_detector(image):#returns array of keypoints
    keypoints = []
    framedImage = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    height, width = image.shape
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            min_ssd = float('inf')
            for windowX, windowY in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                ssd = (float(framedImage[y, x]) - float(framedImage[y + windowY, x + windowX]))**2
                min_ssd = min(min_ssd, ssd)
            threshold = 6000
            if min_ssd > threshold:
                keypoints.append((x - 1, y - 1))        
    return keypoints

# the [y,x] is because y represents the row while x is the cloumn. FOr 2d, in 3d y is the matrix, x is the row.
# a single pixle using this axis (2) is an entire row so we can manipulate its values without changing rows.  
def plot_keypoints(image, keypoints):
    image_with_keypoints = np.stack((image,) * 3, axis=2)
    for keypoint in keypoints:
        x, y = keypoint
        image_with_keypoints[y, x] = [0, 0, 255]
    return image_with_keypoints

def histogramize(featureVector):
    histogram = []
    for x in range(256):
        histogram.append(0)
    for y in range(256):
        histogram[featureVector[y]] += 1
    for z in range(256):
        histogram[z] /= 256
    return histogram

def extract_LBP(imageInput, keypoint):
    image = np.pad(imageInput, ((9, 9), (9, 9)), mode='constant')
    x, y = keypoint
    featureVector = []
    for windowY in range(-7, 9):
        for windowX in range(-7, 9):
            binaryNum = 0
            bitPlace = 1
            for subWindowX, subWindowY in [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1]:
                if image[y+windowY+subWindowY+8, x+windowX+subWindowX+8] >= image[y+windowY+8,x+windowX+8]:#check later
                    binaryNum |= bitPlace
                bitPlace <<= 1
            featureVector.append(binaryNum)

    return histogramize(featureVector)

def extract_HOG(imageInput, keypoint):
    # resizedImage = cv2.resize(image,(64,128))
    height, width = imageInput.shape
    image = np.pad(imageInput, ((8, 8), (8, 8)), mode='constant')
    keyX, keyY = keypoint
    featureVector = [0] * 18
    # featureVector = [[0 for x in range(18)]]#this will be the bins, index 0 is 0-19, 1 is 20-39....
    dataMap1 = [[0 for x in range(width)] for y in range(height)]
    dataMap1 = np.pad(dataMap1, ((8, 8), (8, 8)), mode='constant')
    dataMap2 = dataMap1
    for y in range(1, height + 8):#gathering info for smaller window, the 16x16 will use.
        for x in range(1, width + 8):
            horzGrad = float(image[y][(x-1)]) - float(image[y][(x+1)])
            vertGrad = float(image[(y-1)][x]) - float(image[(y+1)][x])
            magnitude = (horzGrad**2+vertGrad**2)**0.5
            if (horzGrad == 0):#edge case for 1/0
                if vertGrad > 0:
                    orientation = 90
                elif vertGrad < 0:
                    orientation =  270
                else:
                    orientation = 0
            else:
                orientation = (np.arctan(vertGrad/horzGrad))*(180/np.pi)
                if vertGrad >= 0 and horzGrad < 0 or vertGrad < 0 and horzGrad < 0:#handle all 4quads
                    orientation += 180
                elif horzGrad >= 0 and vertGrad < 0:
                    orientation += 360
                else:
                    pass
            dataMap1[y][x] = orientation
            dataMap2[y][x] = magnitude
    #now create historgram for 16x16 window
    for windowY in range(-7, 9):
        for windowX in range(-7, 9):
            orientation = dataMap1[keyY+windowY+8][keyX+windowX+8]
            magnitude = dataMap2[keyY+windowY+8][keyX+windowX+8]
            index = int(orientation//20)#bin size 20
            featureVector[index] += ((((index+1)*20)-orientation)/20)*magnitude#left bin
            if index == 17:
                index == 0
            else:
                index += 1
            featureVector[index] += ((orientation-(index*20))/20)*magnitude#right bin
    for z in range(18):
        featureVector[z] /= 18
    return featureVector

def feature_matching(image1, image2, detector, extractor):
    if detector != "Moravec":# and detector != "Harris":
        raise Exception("Sorry, the detector must be \"Moravec\"")# or \"Harris\"")
    if extractor != "LBP" and extractor != "HOG":
        raise Exception("Sorry, the detector must be \"LBP\" or \"HOG\"")
    if detector == "Moravec":
        moravecedPoints1 = moravec_detector(image1)
        moravecedPoints2 = moravec_detector(image2)
    # else:
    #     keypoints1 = harris_detector(image1)
    #     keypoints2 = harris_detector(image2)
    threshold = 0.75 if extractor == "LBP" else 0
    list1 = []
    list2 = []
    for p1 in (moravecedPoints1):#for every pixle in the array, do to every other pixle in the other array
        hist1 = np.array(extract_LBP(image1, p1) if extractor == "LBP" else extract_HOG(image1, p1))#this will actually make it slower but its less lines of code
        for p2 in (moravecedPoints2):
            hist2 = np.array(extract_LBP(image2, p2) if extractor == "LBP" else extract_HOG(image2, p2))
            res = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            if res >= threshold:#make lists of keypoints from the first image that match the second image
                list1.append(p1)
                list2.append(p2)
                break#this break point is optional, allows for spiderweb look. on is less, off is more
    return (list1, list2)

def plot_matches(image1, image2, matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    max_height = max(image1.shape[0], image2.shape[0])
    total_width = image1.shape[1] + image2.shape[1]
    new_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    new_image[:image1.shape[0], :image1.shape[1]] = image1
    new_image[:image2.shape[0], image1.shape[1]:] = image2
    offset = image1.shape[1]
    matches1, matches2 = matches
    for i in range(len(matches1)):
        kp1 = matches1[i]
        kp2 = matches2[i]
        # Convert keypoints from tuple to cv2.KeyPoint, draws circle
        kp1 = cv2.KeyPoint(kp1[0], kp1[1], 1)
        kp2 = cv2.KeyPoint(kp2[0] + offset, kp2[1], 1)
        cv2.drawKeypoints(new_image, [kp1, kp2], new_image, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        cv2.line(new_image, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), (0, 0, 255), thickness=1)    
    return new_image
