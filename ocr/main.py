from sys import platform
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pytesseract


class Base:
    inverted_img_write = False

    vertical_img_write = False

    horizontal_img_write = False

    horizontal_with_vertical_img_write = False

    # Setup installed path (windows operating system)
    tesseract_setup_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def __init__(self, file, destination = f"E:\laragon\www\demos\php-python-ocr-image-table\\rendered_files\\"):
        # print("init this Base class.")
        self.file = file
        self.destination_path = destination
        self.img = cv2.imread(self.file, 0)
        self.img.shape

    def thresholding_to_binary_image(self):
        thresh, img_bin = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        return (thresh, img_bin)

    def inverting_image(self, is_saw=False):
        # inverting the image
        thresh, img_bin = self.thresholding_to_binary_image()

        if(self.inverted_img_write == True):
            cv2.imwrite(self.destination_path + 'cv_inverted.png', img_bin)
            # Plotting the image to see the output
            plotting = plt.imshow(img_bin, cmap='gray')
            plt.show() if is_saw == True else ''
        return True

    def countcol_of_kernel_as_100_total_width(self):
        # countcol(width) of kernel as 100th of total width
        kernel_len = np.array(self.img).shape[1] // 100
        # Defining a vertical kernel to detect all vertical lines of image
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        # Defining a horizontal kernel to detect all horizontal lines of image
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        # A kernel of 2x2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        return (kernel_len, ver_kernel, hor_kernel, kernel)

    def count_columns_of_kernel_len(self):
        # countcol(width) of kernel as 100th of total width
        return np.array(self.img).shape[1] // 100

    def kernel_of_2x2(self):
        # A kernel of 2x2
        return cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    def vertical_kernel_to_vertical_image(self, is_saw=False):
        thresh, img_bin = self.thresholding_to_binary_image()

        # Defining a vertical kernel to detect all vertical lines of image
        kernel_len, ver_kernel, hor_kernel, kernel = self.countcol_of_kernel_as_100_total_width()

        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

        if(self.vertical_img_write == True):
            cv2.imwrite(self.destination_path + "vertical.jpg", vertical_lines)
            # Plot the generated image
            plotting = plt.imshow(image_1, cmap='gray')
            plt.show() if is_saw == True else ''

        return vertical_lines

    def horizontal_kernel_to_horizontal_image(self, is_saw=False):
        thresh, img_bin = self.thresholding_to_binary_image()

        # Defining a vertical kernel to detect all vertical lines of image
        kernel_len, ver_kernel, hor_kernel, kernel = self.countcol_of_kernel_as_100_total_width()

        # Use horizontal kernel to detect and save the horizontal lines in a jpg
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

        if(self.horizontal_img_write == True):
            cv2.imwrite(self.destination_path + "horizontal.jpg", horizontal_lines)
            # Plot the generated image
            plotting = plt.imshow(image_2, cmap='gray')
            plt.show() if is_saw == True else ''

        return horizontal_lines

    def combine_horizontal_and_vertical_lines_image(self, is_saw=False):

        vertical_lines = self.vertical_kernel_to_vertical_image()
        horizontal_lines = self.horizontal_kernel_to_horizontal_image()

        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        # Eroding and thesholding the image
        img_vh = cv2.erode(~img_vh, self.kernel_of_2x2(), iterations=2)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        bitxor = cv2.bitwise_xor(self.img, img_vh)
        bitnot = cv2.bitwise_not(bitxor)

        if(self.horizontal_with_vertical_img_write == True):
            cv2.imwrite(self.destination_path + "img_vh.jpg", img_vh)
            # Plotting the generated image
            plotting = plt.imshow(bitnot, cmap='gray')
            plt.show() if is_saw == True else ''

        return (img_vh, bitxor, bitnot)

    def detect_contours(self):
        img_vh, bitxor, bitnot = self.combine_horizontal_and_vertical_lines_image()

        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return (img_vh, bitxor, bitnot, contours, hierarchy)

    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    def contours(self):
        img_vh, bitxor, bitnot, contours, hierarchy = self.detect_contours()
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")

        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes)

    def heights(self):
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes = self.contours()
        heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights)

    def mean(self):
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights = self.heights()
        mean = np.mean(heights)
        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean)

    def get_position_x_and_y_width_and_height(self, is_saw=False):
        # Get position (x,y), width and height for every contour and show the contour on image
        box = []
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes = self.contours()
        image = ""
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w < 1000 and h < 500):
                image = cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                box.append([x, y, w, h])

        plotting = plt.imshow(image, cmap='gray')
        plt.show() if is_saw == True else ''
        return box

    def sorting_boxes_to_their_respective_row_and_column(self):
        row = []
        column = []
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean = self.mean()
        box = self.get_position_x_and_y_width_and_height()
        # Sorting the boxes to their respective row and column
        for i in range(len(box)):

            if (i == 0):
                column.append(box[i])
                previous = box[i]
            else:
                if (box[i][1] <= previous[1] + mean / 2):
                    column.append(box[i])
                    previous = box[i]

                    if (i == len(box) - 1):
                        row.append(column)

                else:
                    row.append(column)
                    column = []
                    previous = box[i]
                    column.append(box[i])

        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column)

    def calculating_max_cells(self):
        j = 0
        # calculating maximum number of cells
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column = self.sorting_boxes_to_their_respective_row_and_column()
        countcol = 0
        for i in range(len(row)):
            countcol = len(row[i])
            if countcol > countcol:
                countcol = countcol

        # Retrieving the center of each column
        center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]

        center = np.array(center)
        center.sort()

        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center)

    def final_boxes(self):
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center = self.calculating_max_cells()

        finalboxes = []
        for i in range(len(row)):
            lis = []
            for k in range(countcol):
                lis.append([])
            for j in range(len(row[i])):
                diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
                minimum = min(diff)
                indexing = list(diff).index(minimum)
                lis[indexing].append(row[i][j])
            finalboxes.append(lis)

        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes)

    def image_based_cell_box(self):
        # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes = self.final_boxes()

        j = 0
        outer = []
        if (platform == "win32"):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_setup_path

        for i in range(len(finalboxes)):
            for j in range(len(finalboxes[i])):
                inner = ''
                if (len(finalboxes[i][j]) == 0):
                    outer.append(' ')
                else:
                    for k in range(len(finalboxes[i][j])):
                        y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                     finalboxes[i][j][k][3]
                        finalimg = bitnot[x:x + h, y:y + w]
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                        border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        dilation = cv2.dilate(resizing, kernel, iterations=1)
                        erosion = cv2.erode(dilation, kernel, iterations=2)

                        out = pytesseract.image_to_string(erosion)
                        if (len(out) == 0):
                            out = pytesseract.image_to_string(erosion, config='--psm 3')
                        inner = inner + " " + out
                    outer.append(inner)

        return (img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes, outer)

    def get_ocr_data(self):
        self.inverting_image()

        # Creating a dataframe of the generated OCR list
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes, outer = self.image_based_cell_box()

        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))

        # NOTE : Print data in json in different orient
        pd_orients = ['columns', 'records', 'index', 'split', 'table']
        for pd_orient in pd_orients:
            print(f"################[ orient = {pd_orient} ]##################")
            print(dataframe.to_json(orient=pd_orient))

    def export_to_excel(self):
        self.inverting_image()

        # Creating a dataframe of the generated OCR list
        img_vh, bitxor, bitnot, contours, hierarchy, contours, boundingBoxes, heights, mean, row, column, countcol, center, finalboxes, outer = self.image_based_cell_box()

        # Creating a dataframe of the generated OCR list
        arr = np.array(outer)
        dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
        data = dataframe.style.set_properties(align="left")

        # Converting it in a excel-file
        export_file_name = "output.xlsx"
        data.to_excel(self.destination_path + export_file_name)

        print("Export Successfully")
