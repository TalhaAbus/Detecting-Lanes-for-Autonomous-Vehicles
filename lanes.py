import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # both of our lines are going to have the same vertical coordinates
    y2 = int(y1*(3/5))
# which means that both of our lines will start from the bottom at y1
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intecept(image, lines):
    left_fit = [] # will contain the coordinates of the average lines on the left
    right_fit = [] # will contain the coordinates of the average lines on the left
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) # unpack the elements of the array into four variables
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # fit a first degree polynomial which would simply be a linear function of y is equal to mx+b
        # fit a polynomial of degree 1 to our x and y points, that way we get the parameters of a linear function
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    #print(left_fit_average,'left')  # represents the average slope and y intercept of a single line
    #print(right_fit_average,'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #create a gray scale from the color image

    blur = cv2.GaussianBlur(gray, (5,5),0)
    #applaying a GaussianBlur on a grayscale image with a 5 by 5 kernel, 0=deviation
    # The size of kernel is dependent on specific situations
    # A 5 by 5 kernel is good size for most cases.
    canny = cv2.Canny(blur, 50, 150) #low and high thresholds of 50 and 150
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0),10)
            #3=what color we want the line to be (RGB) 4= line thickness
    return line_image

def region_of_interest(image):
#this function will do is pretty self-explanatory and will return the enclosed
#region of our field of view and recall that the enclosed region was triangular in shape.
    height = image.shape[0]
    polygons = np.array([
    [(200, height ), (1100, height),(550,250)]
    ])
    mask = np.zeros_like(image)
#we're going to apply it on to a black mask with the same dimensions as our road image.
#fill this mask this black image with our polygon using cv.fillPoly function
    cv2.fillPoly(mask, polygons,255) # 0= fill what?, 1= fill with what?, 2 = specifies color of our polygon
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('test_image.jpg')  #setting image is equal to see me to read
# #we're going to work with a copy of this array
# lane_image = np.copy(image) # if we do this only =image ,
# #any changes we make to lane image will also be reflected in the original viewable array
# canny_image = canny(lane_image)
# cropped_image= region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]),
# minLineLength=40, maxLineGap=5) # 0= image where you want to detect lines which would simply be our cropped image
# averaged_lines = average_slope_intecept(lane_image, lines)
# line_image = display_lines(lane_image,averaged_lines)
# combo_image = cv2.addWeighted(lane_image,0.8,line_image, 1,1)
#2= second input array of the same size, multiplying all elemets in this array by 1
# # 1,2 = specify the resolution of the half accumulator array 3= threshold to find and display the line from a series of dots
# #4= minimum number of intersections in space for a bin needs to be 100 for to be accepted as irrelevant line inscribing our data
# #5= length of a line in pixels that we will accept into the output
# #6= indicates the maximum distance in pixels between segmented lines which
#we will allow to be connected into a single line instead of them being broken up.
# # accumulator array: two dimensional array of rows and columns which contain the bins that we're going to use to collect votes.
# cv2.imshow('result', combo_image) #name of the window that we're going to open up, image that we that we want to show itself
# cv2.waitKey(0) #it displays the image for specified amount of miliseconds
# # it will display our window result to window infinitely untiğl we press anything in our keyboard

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):  # this returns true if video capturing has been initialized
#we will enter into a loop where we will first use the read function "cap.read" to decode every video frame
    _, frame = cap.read()# the first value is just a boolean,
    #second is the image the frame that's currently being projected in our video
    # and it is the current frame of our video where we will be detecting lines
    canny_image = canny(frame)
    cropped_image= region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    #0= image where you want to detect lines which would simply be our cropped image
    averaged_lines = average_slope_intecept(frame, lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image, 1,1)
    #2= second input array of the same size, multiplying all elemets in this array by 1
    # 1,2 = specify the resolution of the half accumulator array 3= threshold to find and display the line from a series of dots
    #4= minimum number of intersections in space for a bin needs to be 100 for to be accepted as irrelevant line inscribing our data
    #5= length of a line in pixels that we will accept into the output
    #6=indicates the maximum distance in pixels between
    #segmented lines which we will allow to be connected into a single line instead of them being broken up.
    # accumulator array: two dimensional array of rows and columns which contain the bins that we're going to use to collect votes.
    cv2.imshow('result', combo_image) #name of the window that we're going to open up, image that we that we want to show itself
    if cv2.waitKey(1) == ord('q'): #it displays the image for specified amount of miliseconds
        break
    # it will display our window result to window infinitely untiğl we press anything in our keyboard
cap.release()
cv2.destroyAllWindows()
