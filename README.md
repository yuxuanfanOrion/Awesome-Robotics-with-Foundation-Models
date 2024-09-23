# Awesome-Robotics-with-Foundation Models [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo was forked from [Awesome-Implicit-NeRF-Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics). Thanks for the inspiration! <br>

 <!-- and contains a curative list of **Implicit Representations and NeRF papers relating to Robotics/RL domain**, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision) <br> -->

#### Please feel free to send me [pull requests](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics/blob/main/how-to-PR.md) or [email](mailto:yfan546@connec.hkust-gz.edu.cn) to add papers! <br>

If you find this repository useful, please consider [citing](#citation) and STARing this list. Feel free to share this list with others!

----
## News
- 2024.09.23: Our Survey is ongoing, will be released soon.
- 2024.09.22: The Repo is built up and the first paper is added.


---
## Overview

- [Awesome-Robotics-with-Foundation Models](#awesome-robotics-nerf-robotics-)
      - [Please feel free to send me pull requests or email to add papers! ](#please-feel-free-to-send-me-pull-requests-or-email-to-add-papers-)
  - [Overview](#overview)
  - [Surveys](#surveys)
  - [SLAM](#slam)
  - [Manipulation/RL](#manipulationrl)
  - [Object Reconstruction](#object-reconstruction)
  - [Physics](#physics)
  - [Planning/Navigation](#planningnavigation)
  - [](#)
  - [Citation](#citation)
 
---
## Surveys

<!-- 
* **BundleSDF**: "Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects", *CVPR, 2023*. [[Paper](https://arxiv.org/pdf/2303.14158.pdf)] [[Webpage](https://bundlesdf.github.io/)]
 -->

* **One-step NeRF**: "Marrying NeRF with Feature Matching for One-step Pose Estimation", *ICRA, 2024*. [[Paper](https://arxiv.org/pdf/2404.00891)] [[Short Video](https://www.youtube.com/watch?v=70fgUobOFWo)] [[Website&Code] Coming]

---
## SLAM
* **iSDF**: "Real-Time Neural Signed Distance Fields for Robot Perception", *RSS, 2022*. [[Paper](https://arxiv.org/abs/2204.02296)] [[Pytorch Code](https://github.com/facebookresearch/iSDF)] [[Website](https://joeaortiz.github.io/iSDF/)]

---
## Manipulation/RL

* **GNFactor**: "GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields", *CoRL 2023 Oral Presentation*. [[Paper/PDF](https://arxiv.org/abs/2308.16891)] [[Code](https://github.com/YanjieZe/GNFactor)] [[Website](https://yanjieze.com/GNFactor/)]


* **D<sup>3</sup>Fields**: "D<sup>3</sup>Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation", *arXiv*. [[Paper](https://arxiv.org/abs/2309.16118)] [[Webpage](https://robopil.github.io/d3fields/)] [[Code](https://github.com/WangYixuan12/d3fields)] [[Video](https://youtu.be/yNkIOwAO3GA)]

* **SNeRL**: "Semantic-aware Neural Radiance Fields for Reinforcement Learning", *ICML, 2023*. [[Paper](https://arxiv.org/pdf/2301.11520.pdf)] [[Webpage](https://sjlee.cc/snerl/)]

* **Ditto**: "Building Digital Twins of Articulated Objects from Interaction", *CVPR, 2022*. [[Paper](https://arxiv.org/abs/2202.08227)] [[Pytorch Code](https://github.com/UT-Austin-RPL/Ditto)] [[Website](https://ut-austin-rpl.github.io/Ditto/)]

* **Relational-NDF**: "SE(3)-Equivariant Relational Rearrangement with Neural Descriptor Fields", *CORL 2022*. [[Paper](https://arxiv.org/pdf/2211.09786.pdf)] [[Pytorch Code](https://github.com/anthonysimeonov/relational_ndf)] [[Website](https://anthonysimeonov.github.io/r-ndf/)]

* **Neural Descriptor Fields**: "SE(3)-Equivariant Object Representations for Manipulation", *arXiv*. [[Paper](https://arxiv.org/abs/2112.05124)] [[Pytorch Code](https://github.com/anthonysimeonov/ndf_robot)] [[Website](https://yilundu.github.io/ndf/)]

* **Evo-NeRF**: "Evolving NeRF for Sequential Robot Grasping of Transparent Objects", *CORL 2022*. [[Paper](https://openreview.net/pdf?id=Bxr45keYrf)]  [[Website](https://sites.google.com/view/evo-nerf)]

* **NeRF-RL**: "Reinforcement Learning with Neural Radiance Fields", *arXiv*. [[Paper](https://dannydriess.github.io/papers/22-driess-NeRF-RL-preprint.pdf)]  [[Website](https://dannydriess.github.io/nerf-rl/)]

* **Neural Motion Fields**: "Encoding Grasp Trajectories as Implicit Value Functions", *RSS 2022*. [[Paper](https://arxiv.org/pdf/2206.14854.pdf)]  [[Video](https://youtu.be/B-pEhT1pi-Q)]

* **Grasping Field**: "Learning Implicit Representations for Human Grasps", *3DV 2020*. [[Paper](https://arxiv.org/pdf/2008.04451.pdf)] [[Pytorch Code](https://github.com/korrawe/grasping_field)] [[Video](https://youtu.be/J8x5i1FCgTQ)]

* **Dex-NeRF**: "Using a Neural Radiance Field to Grasp Transparent Objects", *CORL, 2021*. [[Paper](https://arxiv.org/abs/2110.14217)]  [[Website](https://sites.google.com/view/dex-nerf)]

* **NeRF-Supervision**: "Learning Dense Object Descriptors from Neural Radiance Fields", *ICRA, 2022*. [[Paper](https://arxiv.org/abs/2203.01913)] [[Pytorch Code](https://github.com/yenchenlin/nerf-supervision-public)] [[Website](https://yenchenlin.me/nerf-supervision/)]

* **GIGA**: "Synergies Between Affordance and Geometry: 6-DoF Grasp Detection via Implicit Representations", *RSS, 2021*. [[Paper](https://arxiv.org/abs/2104.01542)] [[Pytorch Code](https://github.com/UT-Austin-RPL/GIGA)] [[Website](https://sites.google.com/view/rpl-giga2021)]

* **NeuralGrasps**: "Learning Implicit Representations for Grasps of Multiple Robotic Hands", *CORL, 2022*. [[Paper](https://arxiv.org/abs/2207.02959)] [[Website](https://irvlutd.github.io/NeuralGrasps/)]

* "Real-time Mapping of Physical Scene Properties with an Autonomous Robot Experimenter", *CORL, 2022*. [[Paper](https://arxiv.org/abs/2210.17325)] [[Website](https://ihaughton.github.io/RobE/)]

* **ObjectFolder**: "A Dataset of Objects with Implicit Visual, Auditory, and Tactile Representations"", *CORL, 2021*. [[Paper](https://arxiv.org/pdf/2109.07991.pdf)] [[Pytorch Code](https://github.com/rhgao/ObjectFolder)] [[Website](https://ai.stanford.edu/~rhgao/objectfolder/)]

* **ObjectFolder 2.0**: "A Multisensory Object Dataset for Sim2Real Transfer"", *CVPR, 2022*. [[Paper](https://arxiv.org/pdf/2204.02389.pdf)] [[Pytorch Code](https://github.com/rhgao/ObjectFolder)] [[Website](https://ai.stanford.edu/~rhgao/objectfolder2.0/)]

* "Template-Based Category-Agnostic Instance Detection for Robotic Manipulation"", *RA-L, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9935113)] 

* **NeRF2Real**: "Sim2real Transfer of Vision-guided Bipedal Motion Skills using Neural Radiance Fields"", *arXiv*. [[Paper](https://arxiv.org/pdf/2210.04932.pdf)] [[Website](https://sites.google.com/view/nerf2real/home)]

* **NeRF-Frenemy**: "Co-Opting Adversarial Learning for Autonomy-Directed Co-Design", *RSS Workshop, 2022*. [[Paper](https://imrss2022.github.io/contributions/lewis.pdf)] [[Website](https://progress.eecs.umich.edu/projects/nerf-frenemy/)]

* **CLA-NeRF**: "Category-Level Articulated Neural Radiance Field", *ICRA, 2022*. [[Paper](https://arxiv.org/pdf/2202.00181.pdf)] [[Website](https://weichengtseng.github.io/project_website/icra22/index.html)]

* **VIRDO**: "Visio-tactile Implicit Representations of Deformable Objects", *ICRA, 2022*. [[Paper](https://arxiv.org/pdf/2202.00868.pdf)] [[Website](https://www.mmintlab.com/research/virdo-visio-tactile-implicit-representations-of-deformable-objects/)]

* **VIRDO++:**: "Real-World, Visuo-Tactile Dynamics and Perception of Deformable Objects", *CORL, 2022*. [[Paper](https://arxiv.org/pdf/2210.03701.pdf)] [[Website](https://www.mmintlab.com/virdopp/)]

* **SceneCollisionNet**: "Object Rearrangement Using Learned Implicit Collision Functions", *ICRA, 2021*. [[Paper](https://arxiv.org/pdf/2011.10726.pdf)] [[Website](https://research.nvidia.com/publication/2021-03_object-rearrangement-using-learned-implicit-collision-functions)]

* "RGB-D Local Implicit Function for Depth Completion of Transparent Objects", *CVPR, 2021*. [[Paper](https://arxiv.org/pdf/2104.00622.pdf)] [[Website](https://research.nvidia.com/publication/2021-03_rgb-d-local-implicit-function-depth-completion-transparent-objects)]

* "Learning Models as Functionals of Signed-Distance Fields for Manipulation Planning", *CORL, 2021*. [[Paper](https://openreview.net/pdf?id=FS30JeiGG3h)] [[Video](https://youtu.be/ga8Wlkss7co)]

* **ContactNets**: "Learning Discontinuous Contact Dynamics with Smooth, Implicit Representations", *CORL, 2020*. [[Paper](https://arxiv.org/pdf/2009.11193.pdf)] [[Pytorch Code](https://github.com/DAIRLab/contact-nets)]

* "Learning Implicit Priors for Motion Optimization", *IROS, 2022*. [[Paper](https://arxiv.org/pdf/2204.05369.pdf)] [[Website](https://sites.google.com/view/implicit-priors)]

* **MIRA**: "Mental Imagery for Robotic Affordances", *CORL, 2022*. [[Paper](https://arxiv.org/pdf/2212.06088.pdf)] [[Website](http://yenchenlin.me/mira/)]

* **NiFR**: "Neural Fields for Robotic Object Manipulation from a Single Image", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2210.12126.pdf)]

* **NIFT**: "Neural Interaction Field and Template for Object Manipulation", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2210.10992.pdf)]

* "Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance", "IROS, 2022". [[Paper](https://arxiv.org/pdf/2210.08537.pdf)]

* **GraspNeRF**: "Multiview-based 6-DoF Grasp Detection for Transparent and Specular Objects Using Generalizable NeRF", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2210.06575v1.pdf)]

* **Touching a NeRF**: "Leveraging Neural Radiance Fields for Tactile Sensory Data Generation ", *CORL, 2022*. [[Paper](https://openreview.net/pdf?id=No3mbanRlZJ)]

* **SE(3)-DiffusionFields**: "Learning smooth cost functions for joint grasp and motion optimization through diffusion", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2209.03855.pdf)] [[Pytorch Code](https://github.com/TheCamusean/grasp_diffusion)]

* **Equivariant Descriptor Fields**: "SE(3)-Equivariant Energy-Based Models for End-to-End Visual Robotic Manipulation Learning ", *ICLR, 2023*. [[Paper](https://openreview.net/pdf?id=dnjZSPGmY5O)] 

* **KP-NERF**: "Dynamical Scene Representation and Control with Keypoint-Conditioned Neural Radiance Field", *CASE, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9926555)]

* **ACID**: "Action-Conditional Implicit Visual Dynamics for Deformable Object Manipulation", *RSS, 2022*. [[Paper](https://arxiv.org/pdf/2203.06856.pdf)] [[Pytorch Code](https://github.com/NVlabs/ACID)]

* **TRITON**: "Neural Neural Textures Make Sim2Real Consistent", *CORL, 2022*. [[Paper](https://arxiv.org/pdf/2206.13500.pdf)] [[Website](https://tritonpaper.github.io/)] [[Pytorch Code](https://github.com/TritonPaper/TRITON)]

* "Perceiving Unseen 3D Objects by Poking the Objects", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2302.13375.pdf)] [[Website](https://zju3dv.github.io/poking_perception/)] [[Pytorch Code](https://github.com/zju3dv/poking_perception)]

* "Feature-Realistic Neural Fusion for Real-Time, Open Set Scene Understanding", *ICRA, 2023*. [[Paper](https://arxiv.org/pdf/2210.03043.pdf)] [[Website](https://makezur.github.io/FeatureRealisticFusion/)]

* **CGF**: "Learning Continuous Grasping Function with a Dexterous Hand from Human Demonstrations", *arXiv*. [[Paper](https://arxiv.org/pdf/2203.06856.pdf)] [[Website](https://jianglongye.com/cgf/)]

* **NGDF**: "Neural Grasp Distance Fields for Robot Manipulation", *arXiv*. [[Paper](https://arxiv.org/pdf/2211.02647.pdf)] [[Website](https://sites.google.com/view/neural-grasp-distance-fields?pli=1)] [[Pytorch Code](https://github.com/facebookresearch/NGDF/)] 

* **NCF**: "Neural Contact Fields: Tracking Extrinsic Contact with Tactile Sensing", *arXiv*. [[Paper](https://arxiv.org/pdf/2210.09297.pdf)] [[Pytorch Code](https://github.com/carolinahiguera/NCF)] 

* **SPARTN**: "NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis", *arXiv*. [[Paper](https://arxiv.org/pdf/2301.08556.pdf)] 

* "RGB-Only Reconstruction of Tabletop Scenes for Collision-Free Manipulator Control", *arXiv*. [[Paper](https://arxiv.org/pdf/2210.11668v1.pdf)] [[Website](https://sites.google.com/nvidia.com/ngp-mpc/)]
  
* "Grasp Transfer based on Self-Aligning Implicit Representations of Local Surfaces", *RAL, 2023*. [[Paper](https://arxiv.org/pdf/2308.07807)] [[Code](https://github.com/Fzaero/Grasp-Transfer)] [[Website](https://fzaero.github.io/GraspTransfer/)]

* "A Real2Sim2Real Method for Robust Object Grasping with Neural Surface Reconstruction", *arXiv*. [[Paper](https://arxiv.org/pdf/2210.02685.pdf)] [[Video](https://www.youtube.com/watch?v=TkvAKLsxkSc)]

* **EndoNeRF**: "Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery", *MICCAI, 2022*. [[Paper](https://arxiv.org/pdf/2206.15255.pdf)] [[Pytorch Code](https://github.com/med-air/EndoNeRF)]

* **NFMP**: "Neural Field Movement Primitives for Joint Modelling of Scenes and Motions", *IROS, 2023*. [[Paper](https://arxiv.org/pdf/2308.05040)] [[Code](https://github.com/Fzaero/Neural-Field-Movement-Primitives)] [[Website](https://fzaero.github.io/NFMP/)]
 
---
## Object Reconstruction

* **NTO3D**: "Neural Target Object 3D Reconstruction with Segment Anything", *CVPR, 2024*. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wei_NTO3D_Neural_Target_Object_3D_Reconstruction_with_Segment_Anything_CVPR_2024_paper.pdf)] [[Code](https://github.com/ucwxb/NTO3D)]

* "Self-supervised Neural Articulated Shape and Appearance Models", *CVPR, 2022*. [[Paper](https://arxiv.org/pdf/2205.08525.pdf)] [[Website](https://weify627.github.io/nasam/)]

* **NeuS**: "Learning Neural Implicit Surfacesby Volume Rendering for Multi-view Reconstruction", *Neurips, 2021*. [[Paper](https://arxiv.org/pdf/2106.10689.pdf)] [[Website](https://lingjie0206.github.io/papers/NeuS/)]

* **VolSDF**: "Volume Rendering of Neural Implicit Surfaces", *Neurips, 2021*. [[Paper](https://arxiv.org/pdf/2106.12052.pdf)] [[Pytorch Code](https://github.com/lioryariv/volsdf)]

* **UNISURF**: "Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction", *ICCV, 2021*. [[Paper](https://arxiv.org/pdf/2104.10078.pdf)] [[Website](https://moechsle.github.io/unisurf/)] [[Pytorch Code](https://github.com/autonomousvision/unisurf)]

* **ObjectSDF**: "Object-Compositional Neural Implicit Surfaces", *ECCV, 2022*. [[Paper](https://arxiv.org/pdf/2207.09686.pdf)] [[Website](https://wuqianyi.top/objectsdf/)] [[Pytorch Code](https://github.com/QianyiWu/objsdf)]

* **IDR**: "Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance", *Neurips, 2020*. [[Paper](https://arxiv.org/pdf/2003.09852.pdf)] [[Website](https://lioryariv.github.io/idr/)] [[Pytorch Code](https://github.com/lioryariv/idr)]

* **DVR**: "Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision", *CVPR, 2020*. [[Paper](https://arxiv.org/pdf/1912.07372.pdf)] [[Pytorch Code](https://github.com/autonomousvision/differentiable_volumetric_rendering)]

* **A-SDF**: "Learning Disentangled Signed Distance Functions for Articulated Shape Representation", *ICCV, 2021*. [[Paper](https://arxiv.org/pdf/2104.07645.pdf)] [[Pytorch Code](https://github.com/JitengMu/A-SDF)]

* **CodeNeRF**: "Disentangled Neural Radiance Fields for Object Categories", *ICCV, 2021*. [[Paper](https://arxiv.org/pdf/2109.01750.pdf)] [[Pytorch Code](https://github.com/wbjang/code-nerf)]

* **DeepSDF**: " Learning Continuous Signed Distance Functions for Shape Representation", *CVPR, 2019*. [[Paper](
https://arxiv.org/pdf/1901.05103.pdf)] [[Pytorch Code](https://github.com/facebookresearch/DeepSDF)]

* **Occupancy networks**: " Learning 3d reconstruction in function space", *CVPR, 2019*. [[Paper](https://arxiv.org/pdf/1812.03828.pdf)] [[Website](https://avg.is.mpg.de/publications/occupancy-networks)]

---
## Physics

* "Inferring Hybrid Neural Fluid Fields from Videos", *Neurips, 2023*. [[Paper](https://arxiv.org/pdf/2312.06561.pdf)] [[Pytorch Code](https://github.com/y-zheng18/HyFluid)] [[Website](https://kovenyu.com/hyfluid/)]

* **DANOs**: "Differentiable Physics Simulation of Dynamics-Augmented Neural Objects", *arXiv*. [[Paper](https://arxiv.org/abs/2210.09420)] [[Video](https://youtu.be/Md0PM-wv_Xg)]

* **PAC-NeRF**: "Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification", *ICLR, 23*. [[Paper](https://openreview.net/pdf?id=tVkrbkz42vc)] [[Website](https://sites.google.com/view/PAC-NeRF)] [[Video](https://youtu.be/Md0PM-wv_Xg)] [[Pytorch Code](https://github.com/xuan-li/PAC-NeRF)]

* **NeuPhysics**: "Editable Neural Geometry and Physics from Monocular Videos", *Neurips, 2022*. [[Paper](https://arxiv.org/abs/2210.12352)] [[Pytorch Code](https://github.com/gaoalexander/neuphysics)] [[Website](https://ylqiao.net/publication/2022nerf/)]

* **NeRF-ysics**: "A Differentiable Pipeline for Enriching NeRF-Represented Objects with Dynamical Properties", *ICRA Workshop, 2022*. [[Paper](https://neural-implicit-workshop.stanford.edu/assets/pdf/lecleach.pdf)]

* "Neural Implicit Representations for Physical Parameter Inference from a Single Video", *WACV, 2023*. [[Paper](https://arxiv.org/pdf/2204.14030.pdf)] [[Pytorch Code](https://github.com/florianHofherr/PhysParamInference)] [[Website](https://florianhofherr.github.io/phys-param-inference/)]

* **NeuroFluid**: "Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields", *ICML, 2022*. [[Paper](https://proceedings.mlr.press/v162/guan22a/guan22a.pdf)] [[Pytorch Code](https://github.com/syguan96/NeuroFluid)]

* "Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data", *SIGGRAPH, 2022*. [[Paper](https://rachelcmy.github.io/pinf_smoke/data/paper.pdf)] [[Pytorch Code](https://github.com/RachelCmy/pinf_smoke)][[Website](https://rachelcmy.github.io/pinf_smoke/)] 

---
## Planning/Navigation
* **NeRFlow**: "Neural Radiance Flow for 4D View Synthesis and Video Processing", *ICCV, 2021*. [[Paper](https://arxiv.org/abs/2012.09790)] [[Pytorch Code](https://github.com/yilundu/nerflow)] [[Website](https://yilundu.github.io/nerflow/)] 

* **NeRF-Navigation**: "Vision-Only Robot Navigation in a Neural Radiance World", *ICRA, 2022*. [[Paper](https://mikh3x4.github.io/nerf-navigation/assets/NeRF_Navigation.pdf)] [[Pytorch Code](https://github.com/mikh3x4/nerf-navigation)] [[Website](https://mikh3x4.github.io/nerf-navigation/)] 

* **RNR-Map**: "Renderable Neural Radiance Map for Visual Navigation", *CVPR, 2023*. [[Paper](https://arxiv.org/pdf/2303.00304.pdf)] [[Website](https://obin-hero.github.io/RNRmap/)] 

* "Uncertainty Guided Policy for Active Robotic 3D Reconstruction using Neural Radiance Fields", *RAL, 2022*. [[Paper (https://arxiv.org/pdf/2209.08409.pdf)] [[Website](https://www.vis.xyz/pub/robotic-3d-scan-with-nerf/)] 

* **NeRF-dy**: "3D Neural Scene Representations for Visuomotor Control", *CORL, 2021*. [[Paper](https://arxiv.org/abs/2107.04004)] [[Website](https://3d-representation-learning.github.io/nerf-dy/)] 

* **CompNeRFdyn**: "Learning Multi-Object Dynamics with Compositional Neural Radiance Fields", *arXiv*. [[Paper](https://arxiv.org/pdf/2202.11855.pdf)] [[Website](https://dannydriess.github.io/compnerfdyn/)] 

* **PIFO**: "Deep Visual Constraints: Neural Implicit Models for Manipulation Planning from Visual Input", *arXiv*. [[Paper](https://arxiv.org/pdf/2112.04812.pdf)] [[Website](https://sites.google.com/view/deep-visual-constraints)] 

* "Learning Continuous Environment Fields via Implicit Functions", *ICLR, 2022*. [[Paper](https://arxiv.org/pdf/2111.13997.pdf)] [[Website](https://research.nvidia.com/publication/2022-04_learning-continuous-environment-fields-implicit-functions)] 

* "Learning Barrier Functions with Memory for Robust Safe Navigation", *RA-L, 2021*. [[Paper](https://arxiv.org/pdf/2011.01899.pdf)]

* **RedSDF**: "Regularized Deep Signed Distance Fields for Reactive Motion Generation", *IROS, 2022*. [[Paper](https://arxiv.org/abs/2203.04739)] [[Website](https://irosalab.com/2022/02/28/redsdf/)] 

* **AutoNeRF**: "Training Implicit Scene Representations with Autonomous Agents", *arxiv*. [[Paper](https://arxiv.org/pdf/2304.11241.pdf)] [[Website](https://pierremarza.github.io/projects/autonerf/)]

* **ESDF**: "Sampling-free obstacle gradients and reactive planning in Neural Radiance Fields", *arXiv*. [[Paper](https://arxiv.org/abs/2205.01389)]

* **CLIP-Fields**: "Open-label semantic navigation with pre-trained VLMs and language models", *arxiv*. [[Paper](https://arxiv.org/abs/2210.05663)] [[Pytorch Code and Tutorials](https://github.com/notmahi/clip-fields)] [[Website](https://mahis.life/clip-fields/)]

* **Voxfield**: Non-Projective Signed Distance Fields for Online Planning and 3D Reconstruction", *IROS, 2022*. [[Paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2022iros.pdf)] [[Pytorch Code](https://github.com/VIS4ROB-lab/voxfield)]

* **Voxblox**: Incremental 3D Euclidean Signed Distance Fields for On-Board MAV Planning, *IROS, 2017*. [[Paper](https://arxiv.org/pdf/1611.03631.pdf)]

* **NFOMP**: Neural Field for Optimal Motion Planner of Differential Drive Robots With Nonholonomic Constraints", *RA-L, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9851532)] [[Video](https://youtu.be/beEfEOmxJfM)]

* **CATNIPS**: Collision Avoidance Through Neural Implicit Probabilistic Scenes", *arXiv*. [[Paper](https://arxiv.org/pdf/2302.12931.pdf)] 

* **MeSLAM**: Memory Efficient SLAM based on Neural Fields, *IEEE SMC, 2022*. [[Paper](https://arxiv.org/pdf/2209.09357.pdf)] 

* **NTFields**: "Neural Time Fields for Physics-Informed Robot Motion Planning", *ICLR, 2023*. [[Paper](https://arxiv.org/pdf/2210.00120.pdf)]

* "Real-time Semantic 3D Reconstruction for High-Touch Surface Recognition for Robotic Disinfection", *IROS, 2022*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9981300)] 

* **NeurAR**: "Neural Uncertainty for Autonomous 3D Reconstruction", *RA-L, 2023*. [[Paper](https://arxiv.org/pdf/2207.10985.pdf)] 

* **IR-MCL**: "Implicit Representation-Based Online Global Localization", *RA-L, 2023*. [[Paper](https://arxiv.org/pdf/2210.03113.pdf)] [[Pytorch Code](https://github.com/PRBonn/ir-mcl)] 

* **360Roam**: "Real-Time Indoor Roaming Using Geometry-Aware 360◦ Radiance Fields", *arXiv*. [[Paper](https://arxiv.org/pdf/2208.02705.pdf)] [[Pytorch Code](https://github.com/PRBonn/ir-mcl)] 

* "Learning Deep SDF Maps Online for Robot Navigation and Exploration", *arXiv*. [[Paper](https://arxiv.org/pdf/2207.10782.pdf)]

* **DroNeRF**: Real-time Multi-agent Drone Pose Optimization for Computing Neural Radiance Fields. [[Paper](https://arxiv.org/pdf/2303.04322.pdf)]

* "Enforcing safety for vision-based controllers via Control Barrier Functions and Neural Radiance Fields", *arXiv*. [[Paper](https://arxiv.org/pdf/2209.12266.pdf)]

* "Full-Body Visual Self-Modeling of Robot Morphologies", *arXiv*. [[Paper](https://arxiv.org/abs/2205.01389)] [[Website](https://huajianup.github.io/research/360Roam/)]

* "Efficient View Path Planning for Autonomous Implicit Reconstruction", *arxiv*. [[Paper](https://arxiv.org/abs/2210.05129)]

* "Multi-Object Navigation with dynamically learned neural implicit representations", *arxiv*. [[Paper](https://arxiv.org/pdf/2209.13159.pdf)] [[Website](https://small-zeng.github.io/EVPP/)

----
[![Star History Chart](https://api.star-history.com/svg?repos=zubair-irshad/Awesome-Implicit-NeRF-Robotics&type=Date)](https://star-history.com/#zubair-irshad/Awesome-Implicit-NeRF-Robotics&Date)
----

## Citation
If you find this repository useful, please consider citing this list:
```
@misc{irshad2022implicitnerfroboticsresources,
    title = {Awesome Implicit NeRF Robotics - A curated list of resources on implicit neural representations and nerf relating to robotics},
    author = {Muhammad Zubair Irshad},
    journal = {GitHub repository},
    url = {https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics},
    DOI= {10.5281/ZENODO.7552613}
    year = {2022},
}
```
