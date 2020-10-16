# Consensus equilibrium framework for subsurface imaging
## What is Consensus Equilibrium?
- Consensus equilibrium (CE) framework was developed to integrate multiple heterogeneous models, and it can be viewed as the generalization of ADMM-based plug-and-play approach.
- The main advantage of CE framework is that we can utilize multiple deep learning based models as well as physical models simultaneously to reconstruct the single image. 
- Previous researches have shown that CE improves the quality of image restoration significantly. 
- Here, we apply CE framework for subsurface image restoration. We propose three different agents for high quality image reconstruction. 
## Three different models for subsurface image reconstruction 
### Data fidelity agent <img src="https://render.githubusercontent.com/render/math?math=F_{data}">
- It is introduced to reduce the mismatch between the observed data and forward simulation of the reconstructed model. 
- It is defined as a proximal mapping of data fidelity function,
<a href="https://www.codecogs.com/eqnedit.php?latex=F_{data}&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;\frac{1}{2}(g(v)&space;-&space;d_{obs})^TC_D^{-1}(g(v)&space;-&space;d_{obs})&space;\big\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{data}&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;\frac{1}{2}(g(v)&space;-&space;d_{obs})^TC_D^{-1}(g(v)&space;-&space;d_{obs})&space;\big\}" title="F_{data} = {\arg\min}_v \big\{ \frac{1}{2\sigma^2}||v-x||^2 + \frac{1}{2}(g(v) - d_{obs})^TC_D^{-1}(g(v) - d_{obs}) \big\}" /></a>
### Denoiser agent <img src="https://render.githubusercontent.com/render/math?math=F_{denoiser}">
- To restore geological structure, we introduce image denoiser as an agent. 
- Existing state-of-the-art denoisers including model-based methods (e.g., total variation (TV) denoiser, BM3D denoiser, etc. ) and learning based methods (e.g., DnCNN denoiser) are deployed as our denoiser agent.
### VAE agent (incorporate qualitative knowledge) <img src="https://render.githubusercontent.com/render/math?math=F_{geology}">
- VAE agent enforces our model to preserve the prior qualitative information such as shapes, sizes, positions and orientations of objects.
- To encapsulate this qualitative knowledge to our model, we propose to use variational autoencoders (VAEs), one of the popular generative models.
- Below figure shows the overall architecture of our VAE designed for learning prior geological information. 
<img src="https://user-images.githubusercontent.com/72419213/96280818-4c4f4180-0f8d-11eb-986f-4118e0e9256a.png" width="500">

- After pretraining the VAE model using prior geological realization, we define VAE agent as 
<a href="https://www.codecogs.com/eqnedit.php?latex=F_{VAE}(x)&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;L_{VAE}(v)&space;\big\}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{geology}(x)&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;L_{VAE}(v)&space;\big\}." title="F_{VAE}(x) = {\arg\min}_v \big\{ \frac{1}{2\sigma^2}||v-x||^2 + L_{VAE}(v) \big\}." /></a>

- Below figure shows how VAE agent operates for two different realizations. 
<img src="https://user-images.githubusercontent.com/72419213/96282456-a0f3bc00-0f8f-11eb-8c60-cf2d671e0ea1.png" width="500">

### Experiment results
- We performed three numerical experiments for restoring the subsurface image to demonstrate the performance of the proposed method. 
- The proposed CE framework employing three different agents yields more realistic subsurface image compare to existing methods. 
- Below figure shows the one simple experiment result.
<img src="https://user-images.githubusercontent.com/72419213/96280813-4b1e1480-0f8d-11eb-807c-d66490d3d295.png" width="500">
