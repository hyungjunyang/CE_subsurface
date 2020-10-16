# Consensus equilibrium framework for subsurface imaging
## What is Consensus Equilibrium?
Consensus equilibrium (CE) framework was developed to integrate multiple heterogeneous models, and it can be viewed as the generalization of ADMM-based plug-and-play approach.
Here, we propose three different agents for subsurface image restoration and integrate them within CE framework 
## Three different models for subsurface image reconstruction 
### Data fidelity agent <img src="https://render.githubusercontent.com/render/math?math=F_{data}">
- It is introduced to reduce the mismatch between the observed data and forward simulation of the reconstructed model. 
- It is defined as a proximal mapping of data fidelity function,
<a href="https://www.codecogs.com/eqnedit.php?latex=F_{data}&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;\frac{1}{2}(g(v)&space;-&space;d_{obs})^TC_D^{-1}(g(v)&space;-&space;d_{obs})&space;\big\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_{data}&space;=&space;{\arg\min}_v&space;\big\{&space;\frac{1}{2\sigma^2}||v-x||^2&space;&plus;&space;\frac{1}{2}(g(v)&space;-&space;d_{obs})^TC_D^{-1}(g(v)&space;-&space;d_{obs})&space;\big\}" title="F_{data} = {\arg\min}_v \big\{ \frac{1}{2\sigma^2}||v-x||^2 + \frac{1}{2}(g(v) - d_{obs})^TC_D^{-1}(g(v) - d_{obs}) \big\}" /></a>
### Denoiser agent <img src="https://render.githubusercontent.com/render/math?math=F_{denoiser}">
- To restore geological structure, we introduce image denoiser as an agent. 
- Existing state-of-the-art denoisers including model-based methods (e.g., total variation (TV) denoiser, BM3D denoiser, etc. ) and learning based methods (e.g., DnCNN denoiser) are deployed as our denoiser agent.
### Geology agent <img src="https://render.githubusercontent.com/render/math?math=F_{geology}">
- Geology agent enforces our model to preserve the prior geological information such as shapes, sizes, positions and orientations of geological objects.
- To encapsulate this geological information to our model, we propose to use variational autoencoders (VAEs), one of the popular generative models.
