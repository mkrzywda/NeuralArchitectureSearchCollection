# Neural Architecture Search (NAS) Papers & Code Collection

## Content

- [Gradient-based Methods](#gradient-based-methods)
  - [CNN](#cnn)
  - [GAN](#gan)
  - [GCN](#gcn)
  - [Mobile](#mobile)
  - [Domain](#domain)
  - [Dilation](#dilation)
  - [Loss](#loss)
  - [DARTS](#darts)
- [Reinforcement Learning (RL) Methods](#reinforcement-learning-rl-methods)
  - [CNN](#cnn-1)
- [Evolutionary Algorithm (EA) Methods](#evolutionary-algorithm-ea-methods)
  - [CNN](#cnn-2)
  - [GCN](#gcn-1)
  - [Mobile](#mobile-1)
- [Bayesian Methods](#bayesian-methods)
  - [CNN](#cnn-3)
- [Other Methods](#other-methods)
  - [CNN](#cnn-4)
  - [Others](#others)
- [Benchmarks](#benchmarks)

## Gradient-based Methods <a name="gradient-based-methods"></a>

### CNN <a name="cnn"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [HR-NAS: Searching Efficient High-Resolution Neural Architectures With Lightweight Transformers](https://arxiv.org/pdf/2106.06560v1.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/dingmyu/HR-NAS) | 2021 |
| [Global2Local: Efficient Structure Search for Video Action Segmentation](https://arxiv.org/pdf/2101.00910v2.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/ShangHua-Gao/G2L-search) | 2021 |
| [ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_ViPNAS_Efficient_Video_Pose_Estimation_via_Neural_Architecture_Search_CVPR_2021_paper.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/luminxu/ViPNAS) | 2021 |
| [LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search](https://arxiv.org/pdf/2104.14545.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/researchmm/LightTrack) | 2021 |
| [One-Shot Neural Ensemble Architecture Search by Diversity-Guided Search Space Shrinking](https://arxiv.org/pdf/2010.00969v3.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/researchmm/NEAS) | 2021 |
| [DOTS: Decoupling Operation and Topology in Differentiable Architecture Search](https://arxiv.org/pdf/2010.00969v3.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/guyuchao/DOTS) | 2021 |
| [RETHINKING ARCHITECTURE SELECTION IN DIFFERENTIABLE NAS](https://openreview.net/pdf?id=PKubaeJkw3) | ICLR 2021 | Gradient | [Code](https://github.com/ruocwang/darts-pt) | 2021 |
| [Geometry-Aware Gradient Algorithms for Neural Architecture Search](https://openreview.net/forum?id=MuSYkd1hxRP) | ICLR 2021 | Gradient | [Code](https://github.com/liamcli/gaea_release) | 2021 |
| [Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets](https://openreview.net/forum?id=rkQuFUmUOg3) | ICLR 2021 | Gradient | [Code](https://github.com/HayeonLee/MetaD2A) | 2021 |
| [SEDONA: Search for Decoupled Neural Networks toward Greedy Block-wise Learning](https://openreview.net/forum?id=XLfdzwNKzch) | ICLR 2021 | Gradient | [Code](https://github.com/mjpyeon/sedona) | 2021 |
| [Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective](https://openreview.net/forum?id=Cnon5ezMHtu) | ICLR 2021 | Gradient | [Code](https://github.com/VITA-Group/TENAS) | 2021 |
| [DrNAS: Dirichlet Neural Architecture Search](https://openreview.net/forum?id=9FWas6YbmB3) | ICLR 2021 | Gradient | [Code](https://github.com/xiangning-chen/DrNAS) | 2021 |
| [Zero-Cost Proxies for Lightweight NAS](https://openreview.net/forum?id=0cmMMy8J5q) | ICLR 2021 | Gradient | [Code](https://github.com/SamsungLabs/zero-cost-nas) | 2021 |
| [MixSearch: Searching for Domain Generalized Medical Image Segmentation Architectures](https://arxiv.org/pdf/2102.13280v1.pdf) | TMI 2021 | Gradient | [Code](https://github.com/lswzjuer/NAS-WDAN) | 2021 |
| [MiLeNAS: Efficient Neural Architecture Search via Mixed-Level Reformulation](https://arxiv.org/pdf/2003.12238.pdf) | CVPR 2020 | Gradient | [Code](https://github.com/chaoyanghe/MiLeNAS) | 2020 |
| [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) | CVPR 2020 | Gradient | [Code](https://github.com/gmh14/RobNets) | 2020 |
| [FedNAS: Federated Deep Learning via Neural Architecture Search](https://chaoyanghe.com/publications/FedNAS-CVPR2020-NAS.pdf) | CVPR 2020 | Gradient | [Code](https://github.com/chaoyanghe/FedNAS) | 2020 |
| [Block-wisely Supervised Neural Architecture Search with Knowledge Distillation](https://www.xiaojun.ai/papers/CVPR2020_04676.pdf) | CVPR 2020 | Gradient | [Code](https://github.com/changlin31/DNA) | 2020 |
| [Densely Connected Search Space for More Flexible Neural Architecture Search](https://arxiv.org/abs/1906.09607) | CVPR 2020 | Gradient | [Code](https://github.com/JaminFong/DenseNAS) | 2020 |
| [Understanding Architectures Learnt by Cell-based Neural Architecture Search](https://openreview.net/pdf?id=H1gDNyrKDS) | ICLR 2020 | Gradient | [Code](https://github.com/automl/RobustDARTS) | 2020 |
| [Once for All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/forum?id=HylxE1HKwS) | ICLR 2020 | Gradient | [Code](https://github.com/mit-han-lab/once-for-all) | 2020 |
| [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/) | CVPR 2019 | Gradient | [Code](https://github.com/D-X-Y/NAS-Projects) | 2019 |
| [CAP: A Context-Aware Neural Predictor for NAS](https://arxiv.org/pdf/2406.02056v1) | - | - | [Code](https://github.com/jihan4431/CAP) | 2024 |
| [Neural Architecture Search for Compressed Sensing Magnetic Resonance Image Reconstruction](https://arxiv.org/abs/2002.09625) | - | - | [Code](https://github.com/yjump/NAS-for-CSMRI) | 2020 |
| [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://arxiv.org/abs/2001.02525) | ICLR 2020 | -  | [Code](https://github.com/JaminFong/FNA) | 2020 |
| [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985) | CVPR 2019 | - | [Code](https://github.com/MenghaoGuo/AutoDeeplab) | 2019 |

### GAN <a name="gan"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search](https://arxiv.org/abs/2007.09180) | 2020 | - | [Code](https://github.com/Yuantian013/E2GAN) | 2020 |
| [AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks](https://arxiv.org/abs/2006.08198) | ICML 2020 | - | [Code](https://github.com/VITA-Group/AGD) | 2020 |
| [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/abs/1908.03835) | ICCV 2019 | - | [Code](https://github.com/VITA-Group/AutoGAN) | 2019 |
| [Searching towards Class-Aware Generators for Conditional Generative Adversarial Networks](https://arxiv.org/abs/2006.14208) | - | - | [Code](https://github.com/PeterouZh/NAS_cGAN) | 2020 |
| [AlphaGAN: Fully Differentiable Architecture Search for Generative Adversarial Networks](https://arxiv.org/abs/2006.09134) | - | - | [Code](https://github.com/yuesongtian/AlphaGAN) | 2020 |
| [GAN Compression: Efficient Architectures for Interactive Conditional GAN](https://arxiv.org/abs/2003.08936) | - | - | [Code](https://github.com/mit-han-lab/gan-compression) | 2020 |
| [Discovering Neural Wirings](https://arxiv.org/pdf/1906.00586.pdf) | NeurIPS 2019 | Gradient | [Code](https://github.com/allenai/dnw) | 2019 |


### GCN <a name="gcn"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Learning GCN for Skeleton-based Human Action Recognition by Neural Searching](https://arxiv.org/abs/1911.04131) | AAAI 2020 | Gradient | [Code](https://github.com/xiaoiker/GCN-NAS) | 2020 |
| [Graph Neural Architecture Search](https://www.researchgate.net/profile/Chuan_Zhou5/publication/342789484_Graph_Neural_Architecture_Search/links/5f0be495299bf18816197d15/Graph-Neural-Architecture-Search.pdf) | IJCAI 2020 | - | [Code](https://github.com/GraphNAS/GraphNAS) | 2020 |
| [A Generic Graph-based Neural Architecture Encoding Scheme for Predictor-based NAS](https://arxiv.org/abs/2004.01899) | ECCV 2020 | - | [Code](https://github.com/walkerning/aw_nas) | 2020 |

### Mobile <a name="mobile"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Neural Architecture Search for Lightweight Non-Local Networks](https://arxiv.org/abs/2004.01961) | CVPR 2020 | Gradient | [Code](https://github.com/meijieru/yet_another_mobilenet_series) | 2020 |

### Domain <a name="domain"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [In Search of Lost Domain Generalization](https://openreview.net/forum?id=lQdXeXDoWtI) | ICLR 2021 | Gradient | [Code](https://github.com/facebookresearch/DomainBed) | 2021 |

### Dilation <a name="dilation"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Inception Convolution With Efficient Dilation Search](https://arxiv.org/pdf/2012.13587v2.pdf) | CVPR 2021 | Gradient | [Code](https://github.com/yifan123/IC-Conv) | 2021 |

### Loss <a name="loss"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Auto Seg-Loss: Searching Metric Surrogates for Semantic Segmentation](https://openreview.net/forum?id=MJAqnaC2vO1) | ICLR 2021 | Gradient | [Code](https://github.com/fundamentalvision/Auto-Seg-Loss) | 2021 |

### DARTS <a name="darts"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR 2019 | - | [Code](https://github.com/quark0/darts) | 2019 |
| [IS-DARTS: Stabilizing DARTS through Precise Measurement on Candidate Importance](https://arxiv.org/pdf/2312.12648v1) | - | - | [Code](https://github.com/hy-he/is-darts) | 2023 |
| [DARTS: Double Attention Reference-based Transformer for Super-resolution](https://arxiv.org/pdf/2307.08837v1.pdf) | - | - | [Code](https://github.com/bia006/darts) | 2023 |
| [Robustifying DARTS by Eliminating Information Bypass Leakage via Explicit Sparse Regularization](https://arxiv.org/pdf/2306.06858v1.pdf) | - | - | [Code](https://github.com/chaoji90/sp-darts) | 2023 |
| [DartsReNet: Exploring new RNN cells in ReNet architectures](https://arxiv.org/pdf/2304.05838v1.pdf) | - | - | [Code](https://github.com/brian-moser/dartsrenet) | 2023 |
| [Operation-level Progressive Differentiable Architecture Search](https://arxiv.org/pdf/2302.05632v1.pdf) | - | - | [Code](https://github.com/zhuxunyu/OPP-DARTS) | 2023 |
| [β-DARTS++: Bi-level Regularization for Proxy-robust Differentiable Architecture Search](https://arxiv.org/pdf/2301.06393v1.pdf) | - | - | [Code](https://github.com/Sunshine-Ye/Beta-DARTS) | 2023 |
| [Pseudo-Inverted Bottleneck Convolution for DARTS Search Space](https://arxiv.org/pdf/2301.01286v3.pdf) | - | - | [Code](https://github.com/mahdihosseini/pibconv) | 2022 |
| [Revisiting Training-free NAS Metrics](https://arxiv.org/pdf/2211.08666v1.pdf) | - | - | [Code](https://github.com/taoyang1122/revisit_trainingfree_nas) | 2022 |
| [NAR-Former: Neural Architecture Representation Learning towards Holistic Attributes Prediction](https://openaccess.thecvf.com//content/CVPR2023/papers/Yi_NAR-Former_Neural_Architecture_Representation_Learning_Towards_Holistic_Attributes_Prediction_CVPR_2023_paper.pdf) | CVPR 2023 | - | [Code](https://github.com/yuny220/nar-former) | 2022 |
| [Λ-DARTS: Mitigating Performance Collapse by Harmonizing Operation Selection among Cells](https://arxiv.org/pdf/2210.07998v2.pdf) | - | - | [Code](https://github.com/dr-faustus/lambda-darts) | 2022 |
| [Generalizing Few-Shot NAS with Gradient Matching](https://openreview.net/pdf?id=_jMtny3sMKU) | ILCR 2022 | Gradient | [Code](https://github.com/skhu101/GM-NAS) | 2022 |
| [AGNAS: Attention-Guided Micro- and Macro-Architecture Search](https://proceedings.mlr.press/v162/sun22a/sun22a.pdf) | - | - | [Code](https://github.com/Sunzh1996/AGNAS) | 2022 |
| [Neural Architecture Search For LF-MMI Trained Time Delay Neural Networks](https://arxiv.org/pdf/2201.03943v3.pdf) | - | - | [Code](https://github.com/skhu101/tdnn-f_nas) | 2022 |
| [DropNAS: Grouped Operation Dropout for Differentiable Architecture Search](https://arxiv.org/pdf/2201.11679v1.pdf) | - | - | [Code](https://github.com/wiljohnhong/dropnas) | 2022 |
| [β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search](http://openaccess.thecvf.com//content/CVPR2022/papers/Ye_b-DARTS_Beta-Decay_Regularization_for_Differentiable_Architecture_Search_CVPR_2022_paper.pdf) | - | - | [Code](https://github.com/Sunshine-Ye/Beta-DARTS) | 2022 |
| [Implantable Adaptive Cells: differentiable architecture search to improve the performance of any trained U-shaped network](https://arxiv.org/pdf/2405.03420v1.pdf) | - | - | [Code](https://gitlab.com/emil-benedykciuk/u-net-darts-tensorflow) | 2024 |
| [FR-NAS: Forward-and-Reverse Graph Predictor for Efficient Neural Architecture Search](https://arxiv.org/abs/2404.15622) | - | Gradient | [Code](https://github.com/emi-group/fr-nas) | 2024 |

## Reinforcement Learning (RL) Methods <a name="reinforcement-learning-rl-methods"></a>

### CNN <a name="cnn-1"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070) | CVPR 2020 | RL | [Code](https://github.com/google/automl/tree/master/efficientdet) | 2020 |
| [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/pdf?id=HylVB3AqYm) | ICLR 2019 | RL/Gradient | [Code](https://github.com/MIT-HAN-LAB/ProxylessNAS) | 2019 |
| [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | CVPR 2019 | RL | [Code](https://github.com/AnjieZheng/MnasNet-PyTorch) | 2019 |

## Evolutionary Algorithm (EA) Methods <a name="evolutionary-algorithm-ea-methods"></a>

### CNN <a name="cnn-2"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection](https://arxiv.org/pdf/2103.04507v3.pdf) | CVPR 2021 | EA | [Code](https://github.com/VDIGPKU/OPANAS) | 2021 |

### GCN <a name="gcn-1"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Learning GCN for Skeleton-based Human Action Recognition by Neural Searching](https://arxiv.org/abs/1911.04131) | AAAI 2020 | EA | [Code](https://github.com/xiaoiker/GCN-NAS) | 2020 |

### Mobile <a name="mobile-1"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)| ICCV 2019 | EA | [Code](https://github.com/kuan-wang/pytorch-mobilenet-v3) | 2019 |

## Bayesian Methods <a name="bayesian-methods"></a>

### CNN <a name="cnn-3"></a> 

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Interpretable Neural Architecture Search using Bayesian Optimisation with Weisfeiler-Lehman Kernel (NAS-BOWL)](https://openreview.net/pdf?id=j9Rv7qdXjd) | ICLR 2021 | Bayesian | [Code](https://github.com/xingchenwan/nasbowl) | 2021 |
| [Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes](https://ui.adsabs.harvard.edu/abs/2020arXiv200707743G/abstract) | ECCV 2020 | - | [Code](https://github.com/ActiveVisionLab/NUQ) | 2020 |
| [FlexiBO: Cost-Aware Multi-Objective Optimization of Deep Neural Networks](https://arxiv.org/abs/2001.06588) | - | - | [Code](https://github.com/softsys4ai/FlexiBO) | 2020 |

## Other Methods <a name="other-methods"></a>

### CNN <a name="cnn-4"></a> 

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective](https://openreview.net/forum?id=Cnon5ezMHtu) | ICLR 2021 | Training-free | [Code](https://github.com/VITA-Group/TENAS) | 2021 |
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) | ICML 2018 | Other | [Code](https://github.com/google-research/nasbench) | 2018 |

### Others <a name="others"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [Towards modular and programmable architecture search](https://arxiv.org/abs/1909.13404) | NeurIPS 2019 | Other | [Code](https://github.com/negrinho/deep_architect) | 2019 |
| [Deep Active Learning with a Neural Architecture Search](https://arxiv.org/pdf/1811.07579.pdf) | NeurIPS 2019 | Other | [Code](https://github.com/anonygit32/active_inas) | 2019 |
| [Efficient Forward Architecture Search](https://arxiv.org/abs/1905.13360) | NeurIPS 2019 | Other | [Code](https://github.com/microsoft/petridishnn) | 2019 |
| [Path-Level Network Transformation for Efficient Architecture Search](https://arxiv.org/abs/1806.02639) | ICML 2018 | Other | [Code](https://github.com/han-cai/PathLevel-EAS) | 2018 |
| [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/pdf/1708.05344.pdf) | ICLR 2018 | Other | [Code](https://github.com/ajbrock/SMASH) | 2018 |
| [Neural Architecture Search with Bayesian Optimisation and Optimal Transport](https://arxiv.org/pdf/1802.07191.pdf) | NeurIPS 2018 | Bayesian | [Code](https://github.com/kirthevasank/nasbot) | 2018 |
| [Designing Neural Network Architectures using Reinforcement Learning](https://openreview.net/pdf?id=S1c2cvqee) | ICLR 2017 | RL | [Code](https://github.com/bowenbaker/metaqnn) | 2017 |
| [NSGANetV2: Evolutionary Multi-Objective Surrogate-Assisted Neural Architecture Search](https://arxiv.org/abs/2007.10396) | ECCV 2020 | - | [Code](https://github.com/mikelzc1990/nsganetv2) | 2020 |
| [Multi-Objective Neural Architecture Search Based on Diverse Structures and Adaptive Recommendation](https://arxiv.org/abs/2007.02749) | - | - |  [Code](https://github.com/wangcn0201/MoARR) | 2020 |
| [FNA++: Fast Network Adaptation via Parameter Remapping and Architecture Search](https://arxiv.org/abs/2006.12986) |  - |ICLR 2020 | [Code](https://github.com/JaminFong/FNA) |
| [Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection](https://arxiv.org/abs/2003.11818) |  - |CVPR 2020* | [Code](https://github.com/ggjy/HitDet.pytorch) |
| [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://arxiv.org/abs/2001.02525) | - | ICLR 2020 | [Code](https://github.com/JaminFong/FNA) |
| [DetNAS: Neural Architecture Search on Object Detection](https://arxiv.org/abs/1903.10979) | - | NeurIPS 2019 | [Code](https://github.com/megvii-model/DetNAS) | 2019 |

## Benchmarks <a name="benchmarks"></a>

| Paper Title | Conference | Strategy | Code | Date |
|:------------|:----------:|:--------:|:----:|:----:|
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) | ICML 2019 | Benchmark | [Code](https://github.com/google-research/nasbench) | 2019 |
| [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://arxiv.org/abs/2001.00326) | ICLR 2020 | Benchmark | [Code](https://github.com/D-X-Y/AutoDL-Projects) | 2020 |
| [NAS-Bench-301: Benchmarking Neural Architecture Search](https://arxiv.org/abs/2008.09777) | NeurIPS 2020 | Benchmark | [Code](https://github.com/automl/nasbench301) | 2020 |
