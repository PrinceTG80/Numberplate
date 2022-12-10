import streamlit as st
from main import Image_Processing
import config
from io import BytesIO
import cv2
from cv2 import Mat
import pytesseract
import numpy as np
import re
import glob
import numpy as np
from PIL import Image
import re
import json


def main():
    """define the steamlit UI"""
    # st.set_page_config(layout="wide")
    st.header("PyTesseract Image Processing")
    st.subheader("Number Plate Recognition")
    uploaded_file = st.sidebar.file_uploader("Choose an image file")
    def number_plate_detection(img):
        def clean2_plate(plate):
            gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
            _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     pass
            num_contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
            if num_contours:
                contour_area = [cv2.contourArea(c) for c in num_contours]
                max_cntr_index = np.argmax(contour_area)
        
                max_cnt = num_contours[max_cntr_index]
                max_cntArea = contour_area[max_cntr_index]
                x,y,w,h = cv2.boundingRect(max_cnt)
        
                if not ratioCheck(max_cntArea,w,h):
                    return plate,None
        
                final_img = thresh[y:y+h, x:x+w]
                return final_img,[x,y,w,h]
        
            else:
                return plate,None
        
        def ratioCheck(area, width, height):
            ratio = float(width) / float(height)
            if ratio < 1:
                ratio = 1 / ratio
            if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
                return False
            return True
        
        def isMaxWhite(plate):
            avg = np.mean(plate)
            if(avg>=115):
                return True
            else:
                return False
        
        def ratio_and_rotation(rect):
            (x, y), (width, height), rect_angle = rect
        
            if(width>height):
                angle = -rect_angle
            else:
                angle = 90 + rect_angle
        
            if angle>15:
                return False
        
            if height == 0 or width == 0:
                return False
        
            area = height*width
            if not ratioCheck(area,width,height):
                return False
            else:
                return True
        
        
        img2 = cv2.GaussianBlur(img, (5,5), 0)

        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        
        img2 = cv2.Sobel(img2,cv2.CV_8U,1,0,ksize=3)
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()	
        _,img2 = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
        morph_img_threshold = img2.copy()
        cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        num_contours, hierarchy= cv2.findContours(morph_img_threshold,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img2, num_contours, -1, (0,255,0), 1)
        # cv2.imshow("Image of car ",img2)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        
        
        for i,cnt in enumerate(num_contours):
            min_rect = cv2.minAreaRect(cnt)
            if ratio_and_rotation(min_rect):
                x,y,w,h = cv2.boundingRect(cnt)
                plate_img = img[y:y+h,x:x+w]
                if(isMaxWhite(plate_img)):
                    clean_plate, rect = clean2_plate(plate_img)
                    if rect:
                        fg=0
                        x1,y1,w1,h1 = rect
                        x,y,w,h = x+x1,y+y1,w1,h1
                        plate_im = Image.fromarray(clean_plate)
                        text = pytesseract.image_to_string(plate_im, lang='eng')
                        return text

    #Quick sort
    def partition(arr,low,high): 
        i = ( low-1 )         
        pivot = arr[high]    
    
        for j in range(low , high): 
            if   arr[j] < pivot: 
                i = i+1 
                arr[i],arr[j] = arr[j],arr[i] 
    
        arr[i+1],arr[high] = arr[high],arr[i+1] 
        return ( i+1 ) 

    def quickSort(arr,low,high): 
        if low < high: 
            pi = partition(arr,low,high) 
    
            quickSort(arr, low, pi-1) 
            quickSort(arr, pi+1, high)
            
        return arr
    
    #Binary search   
    def binarySearch (arr, l, r, x): 
    
        if r >= l: 
            mid = l + (r - l) // 2
            if arr[mid] == x: 
                return mid 
            elif arr[mid] > x: 
                return binarySearch(arr, l, mid-1, x) 
            else: 
                return binarySearch(arr, mid + 1, r, x) 
        else: 
            return -1
        

    print("HELLO!!")
    print("Welcome to the Security system.\n")

    array=[]

    # dir = "C:/Users/HP/Desktop/Automatic-Number-plate-detection-for-Indian-vehicles-main/Automatic-Number-plate-detection-for-Indian-vehicles-main"

    # for img in glob.glob(dir+"/Dataset/*.jpeg") :
    #     img=cv2.imread(img)
        
    #     img2 = cv2.resize(img, (600, 600))
    #     # cv2.imshow("Image of car ",img2)
    #     # cv2.waitKey(1000)
    #     # cv2.destroyAllWindows()
        
        
    #     number_plate=number_plate_detection(img)
    #     res2 = str("".join(re.split("[^a-zA-Z0-9]*", str(number_plate))))
    #     res2=res2.upper()
    #     print(res2)

    #     array.append(res2)

    if uploaded_file:
        st.sidebar.subheader("Original image")
        st.sidebar.image(uploaded_file)
        image = Image_Processing(uploaded_file)

        # col1, col2 = st.columns(2)
        # with col2:
        #     st.subheader("Thresh settings")
        #     lower = st.slider("Lower bound", 0, 255, 100)
        #     upper = st.slider("Upper bound", 0, 255, 255)
        # with col1:
        #     st.subheader("Thresh")
        #     thresh = image.threshold_img(lower, upper)
        #     st.image(thresh)

        # col3, col4 = st.columns(2)
        # with col4:
        #     st.subheader("Mask settings")
        #     choice_struct = st.radio(
        #         "Structuring element",
        #         config.structuring_element,
        #         index=2,
        #     )
        #     choice_morph = st.radio(
        #         "Morphological operation",
        #         config.morphological_operation,
        #         index=3,
        #     )
        # with col3:
        #     st.subheader("Masked image")
        #     masked = image.mask_img(thresh, choice_struct, choice_morph)
        #     st.image(masked)

        # col5, col6 = st.columns(2)
        # with col6:
        #     st.subheader("Adaptive Threshold settings")
        #     choice_adapt_thresh = st.radio(
        #         "Adaptive Method",
        #         config.adaptive_method,
        #     )
        #     choice_thresh = st.radio(
        #         "Threshold type",
        #         config.threshold_type,
        #     )
        #     block = st.slider("Block size", 1, 99, 61, 2)
        #     constant = st.slider("Constant", 1, 100, 11)
        # with col5:
        #     st.subheader("Adaptive Threshold")
        #     adapt_thresh = image.adaptive_thresh(
        #         masked, choice_adapt_thresh, choice_thresh, block, constant
        #     )
        #     st.image(adapt_thresh)

        # col7, col8 = st.columns(2)
        # with col8:
        #     st.subheader("Dilate settings")
        #     ite = st.slider("Number of iterations", 1, 4)
        #     gauss_blur = st.slider("Gaussian Blur", 1, 15, 1, 2)
        #     size = st.slider("Size of the structuring elements", 1, 40, 10)
        # with col7:
        #     st.subheader("Dilated image")
        #     dilated = image.dilate(adapt_thresh, ite, gauss_blur, size)
        #     st.image(dilated)

        # col9, col10 = st.columns(2)
        # with col10:
        #     st.subheader("Contours settings")
        #     width_min = st.slider("Minimum width", 0, 3000, 100)
        #     width_max = st.slider("Maximum width", 0, 3000, 3000)
        #     height_min = st.slider("Minimum height", 0, 3000, 100)
        #     height_max = st.slider("Maximum height", 0, 3000, 3000)
        # with col9:
        #     st.subheader("Contours detected")
        #     try:
        #         rectangles, rois = image.find_contours(
        #             dilated, width_min, height_min, width_max, height_max
        #         )
        #         st.image(rectangles)
        #     except UnboundLocalError:
        #         st.write("No contour detected")
        #         text = ""

        # col11, col12, col13 = st.columns(3)
        # with col13:
        #     st.subheader("PyTesseract settings")
        #     psm = st.radio(
        #         "Page segmentation modes",
        #         config.page_segmentation_modes,
        #         index=3,
        #     )
        #     lang = st.radio("Language", ("English", "French"))
        # with col11:
        #     with st.expander("Show extracted text"):
        #         text = image.contour_to_text(rois, psm, lang)
        #         if st.button("Save text to file"):
        #             image.save_text_to_file(text)
        #         st.text(text)
        # with col12:
        #     with st.expander("Show individual ROI"):
        #         if st.button("Save individual ROI"):
        #             image.save_image_to_file(rois)
        #         for roi in rois:
        #             x, y, w, h = roi[0]
        #             roi_img = masked[y : y + h, x : x + w]
        #             st.write(roi[1])
        #             st.image(roi_img)
        col14,col15,col16 = st.columns(3)
        with col14:
            with st.expander("Gray Scale image"):
                st.subheader("Gray Scale image")                   
                number_plate,img1,img2,edged,image1,image2,new_img=image.number_plate_detection()
                st.image(img1)
        with col15:
            st.subheader("Smoothened image")                   
            st.image(img2)
        with col16:
            st.subheader("Edged image")                   
            st.image(edged)
        col20,col21,col22 =st.columns(3)
        with col20:
            st.subheader("All Contours drawn image")                   
            st.image(image1)
        with col21:
            st.subheader("TOp 30 contours image")                   
            st.image(image2)
        with col22:
            st.subheader("Focused image")                   
            st.image(new_img)
        col17,col18,col19 = st.columns(3)
        with col17:
            st.subheader("Number Plate")
            res2 = str("".join(re.split("[^a-zA-Z0-9]*", str(number_plate))))
            res2=res2.upper()
            st.text(res2)
            carid = res2
                # print(res2)
        
        with col18:
            st.subheader("Check Payment")
            val = json.load(open("entries.json"))
            if carid in val:
                st.text("The car exists in the data")
                if val[carid]["payment"] == "1":
                    st.text("Car is parked in "+ str(val[carid]["parking"]))
                else:
                    st.text("Please pay for the parking.")
            else:
                st.text("The Vehicle is not Allowed")
            # st.text(val)

        with col19:
            st.subheader("Mark paid")
            val = json.load(open("entries.json"))
            if carid in val:
                result = val[carid]["payment"]
                if(int(result) == 1):
                    st.text("Payment is done")
                else:
                    if st.button("Paid"):
                        val[carid]["payment"] = "1"
                        # Serializing json
                        json_object = json.dumps(val)
                        
                        # Writing to sample.json
                        with open("entries.json", "w") as outfile:
                            outfile.write(json_object)
            else:
                st.text("Contact the admin")


