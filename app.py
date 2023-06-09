import os
import streamlit as st
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from PIL import Image
import imageio as imageio
from io import BytesIO, StringIO
import requests


st.set_page_config(page_title="PCA Image Reconstruction", page_icon=":tada:", layout="wide")
with st.container():
	st.title("PCA Image Reconstruction")

STYLE = """
<style>
	img {
		max-width: 100%;
	}
"""
# Define the CSS style
st.markdown("""
    <style>
        .container {
            display: flex;
            flex-direction: row;
        }

        .column {
            flex: 50%;
            padding: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# IMPORTING IMAGE USING SCIPY AND TAKING R,G,B COMPONENTS


def decomp(img_array, image_2d): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
    mean_ = np.mean(image_2d , axis = 1).reshape((image_2d.shape[0], 1))
    cov_mat = image_2d - mean_
    eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
    p = np.size(eig_vec, axis =1)
    idx = np.argsort(eig_val)
    idx = idx[::-1]
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]

    loss = []
    #############LOSS COMPUTATION###############
    max_comp = min(img_array.shape[0], img_array.shape[1])
    n_components = [i for i in range(1, max_comp+1, 20)]
 
    for k in n_components:
        eigvecs_k = eig_vec[:, :k]
        score = np.dot(eigvecs_k.T, cov_mat)
        eigVec_score = np.dot(eigvecs_k, score)
        recon = eigVec_score + np.mean(image_2d, axis = 1).T.reshape((eigVec_score.shape[0], 1)) # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
        reconstruction_error = np.sum((image_2d - recon_img_mat) ** 2)
        
        # print("Number of components:", k)
        # print("PCA reconstruction loss:", reconstruction_error)
        loss.append(reconstruction_error)

    return p, cov_mat, eig_vec, loss

def recons(a_, no_of_comp, p, cov_mat, image_2d, eig_vec, in_image): 
        numpc = no_of_comp # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
        if numpc <p or numpc >0:
            eig_vec = eig_vec[:, range(numpc)]
        score = np.dot(eig_vec.T, cov_mat)
        eigVec_score = np.dot(eig_vec, score)
        recon = eigVec_score + np.mean(image_2d, axis = 1).T.reshape((eigVec_score.shape[0], 1)) # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES

        bits_per_pixel = a_.dtype.itemsize * a_.shape[-1] * 8
        orig_size = (in_image.size[0]) * (in_image.size[1])#a_ * bits_per_pixel
        print(f"Image size:{in_image.size}")
        reduced_size = (no_of_comp)*(min(in_image.size[0], in_image.size[1])*2) #recon_img_mat * np.log2(no_of_comp+1)

        return recon_img_mat, in_image, orig_size, reduced_size

# """run this function"""
# st.info(__doc__)
st.markdown(STYLE, unsafe_allow_html=True)
file = st.file_uploader("Upload an image with lower size (preferably less than 50 KB)", type=["png", "jpg"])
cont = st.empty()
if not file:
	# cont.info(f"Please Upload an image :{' '.join(['png', 'jpg'])}")
	url = 'https://raw.githubusercontent.com/satyamg1620/Prerequsite_test_22210041/main/947_2000.jpg'
	data = requests.get(url)
	file = BytesIO(data.content)

if isinstance(file, BytesIO):	
	in_image = Image.open(file)
	img_array = imageio.imread(file)
	max_comp = min(img_array.shape[0], img_array.shape[1])


print(img_array.shape)
a_np = np.array(img_array)
a_r = a_np[:,:,0]
a_g = a_np[:,:,1]
a_b = a_np[:,:,2]

p_r, cov_mat_r, eig_vec_r, loss_r = decomp(img_array, a_r)
p_g, cov_mat_g, eig_vec_g, loss_g  = decomp(img_array, a_g)
p_b, cov_mat_b, eig_vec_b, loss_b  = decomp(img_array, a_b)
# with st.sidebar:
slider_comp = st.slider('Slide over to change number of componenets', 0, max_comp, 0)
            
with st.spinner('LOADING'): 

    no_of_comp = slider_comp

    # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
    a_r_recon, in_image, orig_size_r, reduced_size_r = recons(a_r, no_of_comp, p_r, cov_mat_r, a_r, eig_vec_r, in_image)
    a_g_recon, in_image, orig_size_g, reduced_size_g = recons(a_g, no_of_comp, p_g, cov_mat_g, a_g, eig_vec_g, in_image)
    a_b_recon, in_image,  orig_size_b, reduced_size_b = recons(a_b, no_of_comp, p_b, cov_mat_b, a_b, eig_vec_b, in_image)

    recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
    recon_color_img = Image.fromarray(recon_color_img)

    col2 = st.empty()
    recon_color_img = recon_color_img.resize((450, 450))
    new_image = in_image.resize((450, 450))
    n_components = [i for i in range(1, max_comp+1, 20)] 
    plt.plot(n_components, loss_r, 'r')
    plt.plot(n_components, loss_g, 'g')
    plt.plot(n_components, loss_b, 'b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss (MSE) VS Number of Principal Components')
    # fig = plt.figure()
    plt.legend(['Red Channel', 'Green Channel', 'Blue Channel'])

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    im = im.resize((450, 450))
    with st.container():
        st.write(f'Total Principal Components : {min(img_array.shape[0], img_array.shape[1])}')
        #st.write(f'Compression Ratio: {100*(reduced_size_r+reduced_size_g+reduced_size_b)/(orig_size_r+orig_size_g+orig_size_b)}')
        caption_li=['Original Image','All three channels Reconstruction Loss' ,f'Reconstructed Image with {no_of_comp} components']
        images = [new_image, im ,recon_color_img]
        st.image(images, caption=caption_li, width=400)


st.markdown('''Principal component analysis, or PCA, is a dimensionality reduction method that is often 
                used to reduce the dimensionality of large data sets, by transforming a large set of variables 
                into a smaller one that still contains most of the information in the large set.
                We can use PCA for dimensionality reduction for images as well.''')

st.markdown('''In this aplication, we are using PCA dimensionality reduction for Image Reconstruction. We can upload an image, the application will first split 
            the image into the three channels (Blue, Green, and Red) first and then and perform PCA separately on each dataset representing each channel and 
            calculate total number of Principal Components of that image. After calculating the number of components, a slider will be shown with a range 
            from 0 to maximum number of principal components of that image.
            We can use the slider increase or decrease the number of components for generating the Reconstructed Image. Additionaly the appication will also show the plot of 
            Loss VS Principal Number of components for each channel i.e Red, Green and Blue 
             ''')
