try:
	import os
	import streamlit as st
	import matplotlib.pyplot as plt 
	import pandas as pd
	import numpy as np 
	from PIL import Image
	import imageio.v2 as imageio
	from io import BytesIO, StringIO
	import requests
	print("All Modules Loaded")
except Exception as e:
	print(f"Some Modules are missing : {e}")

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
def main(img_array, no_of_comp):
	print(f'Inside main func and type of img_array is {type(img_array)}')
	print(img_array.shape)
	a_np = np.array(img_array)
	a_r = a_np[:,:,0]
	a_g = a_np[:,:,1]
	a_b = a_np[:,:,2]

	def comp_2d(image_2d, no_of_comp): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
		mean_ = np.mean(image_2d , axis = 1).reshape((image_2d.shape[0], 1))
		cov_mat = image_2d - mean_
		eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
		p = np.size(eig_vec, axis =1)
		idx = np.argsort(eig_val)
		idx = idx[::-1]
		eig_vec = eig_vec[:,idx]
		eig_val = eig_val[idx]
		numpc = no_of_comp # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
		if numpc <p or numpc >0:
			eig_vec = eig_vec[:, range(numpc)]
		score = np.dot(eig_vec.T, cov_mat)
		eigVec_score = np.dot(eig_vec, score)
		recon = eigVec_score + np.mean(image_2d, axis = 1).T.reshape((eigVec_score.shape[0], 1)) # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
		recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
		return recon_img_mat

	no_of_comp = no_of_comp
	a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r, no_of_comp), comp_2d(a_g, no_of_comp), comp_2d(a_b, no_of_comp) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
	recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
	# recon_color_img = recon_color_img.transpose()
	recon_color_img = Image.fromarray(recon_color_img)
	return recon_color_img

# """run this function"""
# st.info(__doc__)
st.markdown(STYLE, unsafe_allow_html=True)
file = st.file_uploader("Upload an image", type=["png", "jpg"])
cont = st.empty()
if not file:
	# cont.info(f"Please Upload an image :{' '.join(['png', 'jpg'])}")
	url = 'https://themayanagari.com/wp-content/uploads/2020/10/ms-dhoni-png-image-free-download.jpg'
	data = requests.get(url)
	file = BytesIO(data.content)

if isinstance(file, BytesIO):	
	in_image = Image.open(file)
	img_array = imageio.imread(file)
	max_comp = min(img_array.shape[0], img_array.shape[1])

	slider_comp = 0
	with st.sidebar:
		slider_comp = st.slider('Slide over to change number of componenets', 0, max_comp, 0)
	no_of_comp = slider_comp
	recon_color_img = main(img_array, no_of_comp)
	with st.container():
		st.write(f'Total Principal Components : {min(img_array.shape[0], img_array.shape[1])}')
	col2 = st.empty()
	recon_color_img = recon_color_img.resize((400, 400))
	new_image = in_image.resize((400, 400))

	with st.container():
		# col1, col2 = st.columns(2)
		col1 = cont
		with col1:
			col1.image(new_image, caption='Original Image')
		with col2:
			col2.image(recon_color_img, caption="Resultant Image")




	
	
