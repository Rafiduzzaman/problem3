# app.py

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from train_model import CVAE  # or copy the class definition here

# Load model
device = torch.device("cpu")
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.title("Handwritten Digit Generator (0–9)")

digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate Images"):
    with torch.no_grad():
        y = torch.eye(10)[digit].repeat(5, 1)
        z = torch.randn(5, model.latent_dim)
        samples = model.decode(z, y).view(-1, 1, 28, 28)

        grid = make_grid(samples, nrow=5, normalize=True)
        npimg = grid.permute(1, 2, 0).numpy()

        st.image(npimg, caption=f"Generated Digit: {digit}", width=400)
