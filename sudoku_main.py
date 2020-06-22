import numpy as np
import cv2

from tools import crop, unwarp_image, make_square , blend_with_original
from sudoku import Sudoku

def sudoku_main(img , corners , required_num_in_sol=""):
    # Processes the image and outputs the image with the solved sudoku
    img_shape = img.shape
    output_shape = (img_shape[1] , img_shape[0])
    # We crop out the sudoku and get the info needed to paste it back (matrix)
    img_cropped_sudoku, transformation = crop(img, corners)
    img_cropped_sudoku_copy = img_cropped_sudoku.copy()
    
    gray = cv2.cvtColor(img_cropped_sudoku.copy(), cv2.COLOR_BGR2GRAY)
    sudoku_crop_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)

    transformation_matrix = transformation['matrix']
    transformation_matrix_inv = transformation['matrix_inv']
    original_shape = transformation['original_shape']

    sudoku = digitize_sudoku(img_cropped_sudoku)
    extracted_digits = extracted_grids(sudoku)
    #disp("extracted_digits" , extracted_digits)

    sudoku.predict_grid_num(confidence_threshold=0)
    predicted_unsolved_grid = predicted_input_grid(sudoku)
    #disp("predicted_unsolved_grid" , predicted_unsolved_grid)

    solution,existing_numbers = sudoku.solve(img_cropped_sudoku, approximate=0.95 , required_num_in_sol= required_num_in_sol)
    
    img_sudoku_final = unwarp_image(img_cropped_sudoku, transformation_matrix_inv, original_shape , output_shape )
    img_final = blend_with_original(img, img_sudoku_final)
    return solution,existing_numbers,img_cropped_sudoku_copy ,sudoku_crop_thresh , extracted_digits ,predicted_unsolved_grid, img_cropped_sudoku , img_final , sudoku


def get_sudoku_box(img, draw_contours=False, test=False):
    '''Finds the biggest object in the image and returns its 4 corners (to crop it)'''
    
    topbottom_edges = (0, img.shape[0]-1)
    leftright_edges = (0, img.shape[1]-1)

    # Preprocessing:
    edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(edges, (7, 7), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 2)

    # Get contours:
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        conts = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in conts:
            epsilon = 0.025*cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            if len(cnt) > 3:
                # 4 corners ==> it's a square
                topleft =       min(cnt, key=lambda x: x[0,0]+x[0,1])
                bottomright =   max(cnt, key=lambda x: x[0,0]+x[0,1])
                topright =      max(cnt, key=lambda x: x[0,0]-x[0,1])
                bottomleft =    min(cnt, key=lambda x: x[0,0]-x[0,1])
                corners = (topleft, topright, bottomleft, bottomright)

                # Ignoring noise objects
                noise = False
                for corner in corners:
                    if corner[0][0] in leftright_edges or corner[0][1] in topbottom_edges:
                        noise = True
                if noise is True:
                    continue
                    
            else:
                # not a sudoku
                return None

            #after playing with the numbers , 10000 for me
            if cv2.contourArea(cnt) > 10000:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if draw_contours is True:
                    cv2.drawContours(edges, [box], 0, (0,255,0), 2)

                # Returns the 4 corners of an object with 4+ corners and area of >10k
                return corners

            else:
                return None
    return None

import math
from scipy import ndimage 
#https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
def optimize_digit(img):
    
    gray=img.copy()
    #gray = cv2.resize(255-gray, (28, 28))
    gray = cv2.resize(gray, (28, 28))
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)
    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    def getBestShift(img):
        cy,cx = ndimage.measurements.center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty

    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    return gray

def disp(name , img ):
    cv2.imshow("{}".format(name) , img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


def digitize_sudoku(sudoku_image, test=False):
    gray = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    black_box = np.zeros((28,28,3),np.uint8)
    black_box = cv2.cvtColor(black_box, cv2.COLOR_BGR2GRAY)
    h, w = sudoku_image.shape[0], sudoku_image.shape[1]

    # Sudoku object that will contain all the information
    sudoku = Sudoku.instance()
    
    # Let The borders of the whole grid (4) 
    sudoku_border = 4
    border = 4
    x = w/9
    y = h/9

    for i in range(9):
        for j in range(9):
            # We get the position of each case (simply dividing the image in 9)
            top     = int(round(y*i+border)) 
            left    = int(round(x*j+border)) 
            right   = int(round(x*(j+1)-border))
            bottom  = int(round(y*(i+1)-border)) 
            if i == 0:
                top+=sudoku_border
            if i == 8:
                bottom-=sudoku_border
            if j == 0:
                left+=sudoku_border
            if j == 8:
                right-=sudoku_border

            point = [[[left,  top]],[[right, top]],[[left,  bottom]],[[right, bottom]]]
            square, _ = crop(gray, point)

            grid_square = square.copy()
            contours, _ = cv2.findContours(grid_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(grid_square, contours, -1, (255, 255, 255), 2)
            contours, _ = cv2.findContours(grid_square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_grid_img_position = [top, right, bottom, left]

            if contours:
                conts = sorted(contours, key=cv2.contourArea, reverse=True)
                # seelcting the biggest contour
                cnt = conts[0]
                minarea = x*y*0.04
                if cv2.contourArea(cnt) > minarea:
                    # Cropping out the number from grid
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    minx , miny = max(min(box, key=lambda g: g[0])[0], 0),max(min(box, key=lambda g: g[1])[1], 0)
                    maxx ,maxy = min(max(box, key=lambda g: g[0])[0], int(x)),min(max(box, key=lambda g: g[1])[1], int(y))

                    number_image = square[miny:maxy, minx:maxx]

                    if number_image is None or number_image.shape[0] < 2 or number_image.shape[1] < 2:
                        # If no number in there
                        sudoku.update_grid(black_box, (i, j), num_grid_img_position)
                    else:
  
                        final = number_image.copy()
                        final = cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 7)
                        final = make_square(final)                
                        final = final/255
                        #final = optimize_digit(final)
                        sudoku.update_grid(final, (i, j), num_grid_img_position)
                else:
                    sudoku.update_grid(black_box, (i, j), num_grid_img_position)
            else:
                sudoku.update_grid(black_box, (i, j), num_grid_img_position)
    return sudoku

def extracted_grids(sudoku):
    v=[]
    for i in range(9):
        h=[]
        for j in range(9):
            im=sudoku.grid[i,j].image
            h.append(addBorder(im , 1))
        h_stack = np.hstack((img for img in h))
        v.append(h_stack)
    extracted_num =  np.vstack((i for i in v))
    return extracted_num

def addBorder(im , size):
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = size
    border = cv2.copyMakeBorder(im,top=bordersize,bottom=bordersize,left=bordersize,right=bordersize,
                                borderType=cv2.BORDER_CONSTANT,value=[255,255,255])
    return border
        
def num2kernal(num):
    kernal_text = np.ones((28,28,3),np.uint8)
    digit_num = cv2.putText(kernal_text,str(num), (6,21),fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.65,color=(255,255,255), thickness=1)
    digit_num = cv2.cvtColor(digit_num, cv2.COLOR_BGR2GRAY)
#     digit_num = addBorder(digit_num , 1)
    return digit_num

def predicted_input_grid(sudoku):

    v=[]
    for i in range(9):
        h=[]
        for j in range(9):
            if(sudoku.grid[i,j].number>0):
                h.append(addBorder(num2kernal(sudoku.grid[i,j].number) , 1))
            else:
                black_grid = np.ones((28,28,3),np.uint8)
                h.append(addBorder(cv2.cvtColor(black_grid, cv2.COLOR_BGR2GRAY) , 1))
        h_stack = np.hstack((img for img in h))
        v.append(h_stack)
    extracted_grid =  np.vstack((i for i in v))
    return extracted_grid



def sudoku_crop_solve_save(count , required_num_in_sol="123456789"):
    
    raw_img_path=r"static\img\sudoku\raw_sudoku_{count}.jpg".format(count=str(count))
    base=r"static\img\sudoku"
    img = cv2.imread(raw_img_path)
    
    corners = get_sudoku_box(img , draw_contours=True)
    if corners is not None:
        solution,existing_numbers,cropped_sudoku , sudoku_crop_thresh , extracted_digits ,predicted_unsolved_grid, solved_cropped_sudoku , img_final , sudoku = sudoku_main(img , corners , required_num_in_sol=required_num_in_sol)
        cropped_sudoku_450 = cv2.resize(cropped_sudoku , (450,450))
        solved_cropped_sudoku = cv2.resize(solved_cropped_sudoku , (450,450))
        cv2.imwrite(base+r"\cropped_sudoku_{count}.jpg".format(count=str(count)) , cropped_sudoku_450)
        cv2.imwrite(url_for('static', filename='img\sudoku\cropped_sudoku_{count}.jpg'.format(count=str(count))) , cropped_sudoku_450)
        cv2.imwrite(base+r"\solved_cropped{count}_sudoku_0.jpg".format(count=str(count)) , solved_cropped_sudoku)
        cv2.imwrite(base+r"\img_final.jpg",img_final)

        print(solution)
        return solution , existing_numbers , sudoku , cropped_sudoku
    else:
        return False , False , False , False
    
    
