
<p align="left">
  <a href="https://github.com/JinLi998/CanonObjaverseDataset/blob/master/paper/One-shot%203D%20Object%20Canonicalization%20based%20on%20Geometric%20and%20Semantic%20Consistency.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-lightblue">
  </a>

  <a href="https://jinli998.github.io/One-shot_3D_Object_Canonicalization/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-blue">
  </a>
</p>

# One-shot 3D Object Canonicalization based on Geometric and Semantic Consistency（CVPR highlight 2025）
This project offers the Canonical Objaverse Dataset, created using the methods outlined in the paper "One-shot 3D Object Canonicalization based on Geometric and Semantic Consistency." Additionally, the project provides an Objaverse toolkit that includes functionalities for downloading Objaverse data, converting GLB files to OBJ format, loading canonical poses, and more.



## ✅ TODO List

 - [x] Release COD dataset and toolkit.
 - [x] Release paper.

 ## 📋 Table of content
 1. [💡 Overview](#1)
 2. [📖 Dataset](#2)
 3. [✏️ Usage](#3)
 4.  [🔍 Citation](#4)

 ## 💡Overview <a name="1"></a> 
<p align="center">
    <img src="./images/method.png" width="750"/> <br />
    <em> 
    </em>
</p>

## 📖 Dataset <a name="2"></a> 
<p align="center">
    <img src="./images/dataset.png" width="750"/> <br />
    <em> 
    </em>
</p>

## ✏️ Usage <a name="3"></a> 
### Dataset toolkit <a name="31"></a> 
#### Requirements
The code has been tested with
- python 3.10
- pytorch 1.9.0
- pytorch3d 0.7.5
- open3d 0.14.1

#### Project Structure
```
├── Example
│   ├── load_canonicalData.py # A pipeline that includes downloads, format conversion, and canonicalization
├── Src
│   ├── DataDownload
│   │   ├── Download.py # Download Objaverse Data from internet
│   │   └── ..
│   ├── Glb2Obj
│   │   ├── glb2Obj.py  # Convert GLB files to OBJ format
│   │   └── ..
│   └── Obj2Canonicalization
│   │   ├── canon_obj.py  # Load canonical labels on object canonicalization
│   │   └── ..
│   └── structure
│   │   ├── ..
│   └── utils
│   │   ├── ..
├── data
    ├── CanonicalObjaverseDataset.json  # Canonicalization labels
    └── canon-annotations.json # Category and object's UID
```

#### Data Processing
To test the model, please run:

```
python -m Examples.load_canonicalData --data_name 'test' 
```

To get all the COD datasets, please run:

```
python -m Examples.load_canonicalData --data_name 'canon' 
```

## 🔍 Citation <a name="4"></a> 

```
@inproceedings{jin2025one,
  title={One-shot 3D Object Canonicalization based on Geometric and Semantic Consistency},
  author={Jin, Li and Wang, Yujie and Chen, Wenzheng and Dai, Qiyu and Gao, Qingzhe and Qin, Xueying and Chen, Baoquan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={16850--16859},
  year={2025}
}
```

## Acknowledgement
This work was completed during a visit to the Visual Computing and Learning Laboratory at Peking University. Special thanks to [Prof. Baoquan Chen](https://baoquanchen.info/) and the co-authors for their support. Grateful acknowledgment is extended to [Prof. Pengshuai Wang](https://wang-ps.github.io/), [Prof. Xifeng Gao](https://scholar.google.com/citations?user=wSUVcN0AAAAJ&hl=en), and [Dr. Siyan Dong](https://siyandong.github.io/) for their guidance and discussions. Thanks also to [Kai Ye](https://illusive-chase.github.io/), [Jia Li](lirity1024@outlook.com), and Yuhang He for their assistance.
