from io import BytesIO
import os
import cv2
from cv2 import Mat
import pytesseract
import numpy as np
import re
import glob
import numpy as np
from PIL import Image
import re


# path to tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class Image_Processing:
    """methods to prepare an image using Opencv and extract its text using PyTesseract"""

    def __init__(self, uploaded_file: BytesIO) -> None:
        self.img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    def threshold_img(self, lower: int = 100, upper: int = 255) -> Mat:
        """change the colors of an image between two values."""
        lower = np.array([lower, lower, lower])
        upper = np.array([upper, upper, upper])
        thresh = cv2.inRange(self.img, lower, upper)
        return thresh

    def mask_img(self, image, struct_elem, choice_morph):
        """take an image and return a masked version (background deleted)"""
        struct_elem = getattr(cv2, struct_elem)
        choice_morph = getattr(cv2, choice_morph)
        kernel = cv2.getStructuringElement(struct_elem, (20, 20))
        morph = cv2.morphologyEx(image, choice_morph, kernel)
        masked = cv2.bitwise_and(self.img, self.img, mask=morph)
        return masked

    def adaptive_thresh(
        self, image: Mat, adaptiveMethod, thresholdType, blocksize: int, constant: int
    ):
        """turn an image in B&W and increase its contrast"""
        adaptiveMethod = getattr(cv2, adaptiveMethod)
        thresholdType = getattr(cv2, thresholdType)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptiv_threshold = cv2.adaptiveThreshold(
            gray, 255, adaptiveMethod, thresholdType, blocksize, constant
        )
        return adaptiv_threshold

    def dilate(self, image: Mat, iterations: int, gauss_blur: int, size: int):
        """take an image and transform it in B&W areas that will be used to delimitate rectangles (Region of Interest)"""
        blur = cv2.GaussianBlur(image, (gauss_blur, gauss_blur), 0)
        threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        dilate = cv2.dilate(threshed, kernal, iterations=iterations)
        return dilate

    def find_contours(
        self,
        image: Mat,
        width_min: int,
        height_min: int,
        width_max: int,
        height_max: int,
    ):
        """identify areas of an image and draw its borders. Return the coordinates of each shape"""
        cnts, hierarchy = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        rois = []
        rect_num = 1
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if width_min < w < width_max and height_min < h < height_max:
                rectangle = cv2.rectangle(
                    self.img, (x, y), (x + w, y + h), color=(36, 255, 12), thickness=4
                )
                rois.append([[x, y, w, h], rect_num])
                num = cv2.putText(
                    self.img,
                    str(rect_num),
                    org=(x + 10, y + h - 10),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=2,
                    color=(36, 255, 12),
                    thickness=4,
                )
                rect_num += 1
        return rectangle, rois

    def contour_to_text(self, rois, psm: str, language: str):
        """use Pytesseract to extract text from an image."""
        psm_re = re.compile(r"\d+ ")
        psm = psm_re.match(psm)
        config_psm = "--psm " + psm[0]
        if language == "English":
            lang = "eng"
        else:
            lang = "fra"
        text = ""
        for roi in rois:
            x, y, w, h = roi[0]
            roi_search = self.img[y : y + h, x : x + w]
            text_py = pytesseract.image_to_string(
                roi_search, lang=lang, config=config_psm
            )
            text += f"***ROI n°{roi[1]}***\n"
            text += text_py + "\n"
        return text

    def save_text_to_file(self, text: str) -> None:
        path = os.getcwd() + "/result"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/savetext.txt", "w", encoding="utf_8") as f:
            f.write(text)

    def save_image_to_file(self, rois: Mat) -> None:
        path = os.getcwd() + "/result"
        if not os.path.exists(path):
            os.makedirs(path)
        for roi in rois:
            x, y, w, h = roi[0]
            roi_img = self.img[y : y + h, x : x + w]
            cv2.imwrite(f"{path}/{x}.jpg", roi_img)

    def number_plate_detection(self):
        # def clean2_plate(plate):
        #     gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        #     _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
        #     # if cv2.waitKey(0) & 0xff == ord('q'):
        #     #     pass
        #     num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #     if num_contours:
        #         contour_area = [cv2.contourArea(c) for c in num_contours]
        #         max_cntr_index = np.argmax(contour_area)
        
        #         max_cnt = num_contours[max_cntr_index]
        #         max_cntArea = contour_area[max_cntr_index]
        #         x,y,w,h = cv2.boundingRect(max_cnt)
        
        #         if not ratioCheck(max_cntArea,w,h):
        #             return plate,None
        
        #         final_img = thresh[y:y+h, x:x+w]
        #         return final_img,[x,y,w,h]
        
        #     else:
        #         return plate,None
        
        # def ratioCheck(area, width, height):
        #     ratio = float(width) / float(height)
        #     if ratio < 1:
        #         ratio = 1 / ratio
        #     if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        #         return False
        #     return True
        
        # def isMaxWhite(plate):
        #     avg = np.mean(plate)
        #     if(avg>=115):
        #         return True
        #     else:
        #         return False
        
        # def ratio_and_rotation(rect):
        #     (x, y), (width, height), rect_angle = rect
        
        #     if(width>height):
        #         angle = -rect_angle
        #     else:
        #         angle = 90 + rect_angle
        
        #     if angle>15:
        #         return False
        
        #     if height == 0 or width == 0:
        #         return False
        
        #     area = height*width
        #     if not ratioCheck(area,width,height):
        #         return False
        #     else:
        #         return True
        
        img1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        
        # img2 = cv2.GaussianBlur(img2, (5,5), 0)
        img2 = cv2.bilateralFilter(img1, 11, 17, 17) 
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        edged = cv2.Canny(img2, 100, 200,apertureSize = 3, L2gradient = True) 
        # cv2.imshow("edged image", edged)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image1=self.img.copy()
        cv2.drawContours(image1,cnts,-1,(0,255,0),3)
        # cv2.imshow("contours",image1)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
        image2 = self.img.copy()
        cv2.drawContours(image2,cnts,-1,(0,255,0),3)
        # cv2.imshow("Top 30 contours",image2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        num_plate = "NONE"
        i=7
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4: 
                x,y,w,h = cv2.boundingRect(c) 
                new_img=self.img[y:y+h,x:x+w]
                plate = pytesseract.image_to_string(new_img, lang='eng')
                if(len(str(plate)) > 2):
                    print("plate1: " + plate)
                    num_plate = plate
                    break
                i+=1
                break
        return num_plate,img1,img2,edged,image1,image2,new_img
        # img2 = cv2.GaussianBlur(self.img, (5,5), 0)

        # # cv2.imshow("Image of car ",img2)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()

        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # # cv2.imshow("Image of car ",img2)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()
        
        # img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)
        # # cv2.imshow("Image of car ",img2)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()	
        # _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
        # morph_img_threshold = img2.copy()
        # cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        # # cv2.imshow("Image of car ",img2)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()
        # num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img2, num_contours, -1, (0,255,0), 1)
        # # cv2.imshow("Image of car ",img2)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()
        
        
        # for i,cnt in enumerate(num_contours):
        #     min_rect = cv2.minAreaRect(cnt)
        #     if ratio_and_rotation(min_rect):
        #         x,y,w,h = cv2.boundingRect(cnt)
        #         plate_img = self.img[y:y+h,x:x+w]
        #         if(isMaxWhite(plate_img)):
        #             clean_plate, rect = clean2_plate(plate_img)
        #             if rect:
        #                 fg=0
        #                 x1,y1,w1,h1 = rect
        #                 x,y,w,h = x+x1,y+y1,w1,h1
        #                 plate_im = Image.fromarray(clean_plate)
        #                 text = pytesseract.image_to_string(plate_im, lang='eng')
        #                 return text


# img_rez = cv2.resize(img, None, fx=0.5, fy=0.5)


# cv2.imwrite("receipt_1/receipt_6cnts.jpg", img)

# with open("receipt_1/savetext.txt", "w", encoding="utf_8") as f:
#     f.write(text)
