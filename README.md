# NBR-Net
NBR-Net: A Non-rigid Bi-directional Registration Network for Multi-temporal Remote Sensing Images


Remote sensing image registration is the basis of change detection, environmental monitoring, and image fusion. Under severe appearance differences, feature-based methods have difficulty finding sufficient feature matches to solve the global transformation and tackling the local deformation caused by height undulations and building shadows. By contrast, non-rigid registration methods are more flexible than feature-based matching methods, while often ignoring the reversibility between images, resulting in misalignment and inconsistency. To this end, this paper proposes a non-rigid bi-directional registration network (NBR-Net) to estimate the flow-based dense correspondence for remote sensing images. We first propose an external cyclic registration network to strengthen the registration reversibility and geometric consistency by registering Image A to Image B and then reversely registering back to image A. Second, we design an internal iterative refinement strategy to optimize the rough predicted flow caused by large distortion and view-point difference. Extensive experiments demonstrate that our method shows performance superior to the state-of-the-art models on the multi-temporal satellite image dataset. Furthermore, we attempt to extend our method to heterogeneous remote sensing image registration, which is more common in the real world. Therefore, we test our pre-trained model in a satellite and Unmanned Aerial Vehicle (UAV) image registration task. Due to the cyclic registration mechanism and coarse-to-fine refinement strategy, the proposed approach obtains the best performance on two GPS-denied UAV image datasets.


Should you make use of this work, please cite the paper accordingly:
Y. Xu, J. Li, C. Du and H. Chen, "NBR-Net: A Nonrigid Bidirectional Registration Network for Multitemporal Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5620715, doi: 10.1109/TGRS.2022.3162094.

@ARTICLE{9740686,  
author={Xu, Yingxiao and Li, Jun and Du, Chun and Chen, Hao},  
journal={IEEE Transactions on Geoscience and Remote Sensing},  
title={NBR-Net: A Nonrigid Bidirectional Registration Network for Multitemporal Remote Sensing Images},   
year={2022},  
volume={60},  
number={},  
pages={1-15},  
doi={10.1109/TGRS.2022.3162094}}
