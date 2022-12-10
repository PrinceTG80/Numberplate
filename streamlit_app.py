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
    st.header("Number Plate Recognition")
    st.subheader("PyTesseract Image Processing")
    with st.expander("Project Details"):
        st.text("157 Tushar Gavkhare")
        st.text("116 Pratik Kale")
        st.text("161 Dnyanada Mahajan")
    uploaded_file = st.sidebar.file_uploader("Choose an image file")

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

        col1, col2 = st.columns(2)
        with col2:
            st.subheader("Thresh settings")
            lower = st.slider("Lower bound", 0, 255, 100)
            upper = st.slider("Upper bound", 0, 255, 255)
        with col1:
            st.subheader("Thresh")
            thresh = image.threshold_img(lower, upper)
            st.image(thresh)

        col3, col4 = st.columns(2)
        with col4:
            st.subheader("Mask settings")
            choice_struct = st.radio(
                "Structuring element",
                config.structuring_element,
                index=2,
            )
            choice_morph = st.radio(
                "Morphological operation",
                config.morphological_operation,
                index=3,
            )
        with col3:
            st.subheader("Masked image")
            masked = image.mask_img(thresh, choice_struct, choice_morph)
            st.image(masked)

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
                number_plate,img1,img2,edged,image1,image2,new_img=image.number_plate_detection()
                st.image(img1)
        with col15:
            with st.expander("Smoothened image"):                  
                st.image(img2)
        with col16:
            with st.expander("Edged image"):                
                st.image(edged)
        col20,col21,col22 =st.columns(3)
        with col20:
            with st.expander("All Contours drawn image"):                   
                st.image(image1)
        with col21:
            with st.expander("TOp 30 contours image"):                
                st.image(image2)
        with col22:
            with st.expander("Focused image"):             
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


