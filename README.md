# Rule-based Automatic Diagnosis of Thyroid Nodules from Intraoperative Frozen Sections using Deep Learning
Frozen sections provide a basis for rapid intraoperative diagnosis that can guide surgery, but the diagnoses often challenge pathologists. Here we propose a rule-based system to differentiate thyroid nodules from intraoperative frozen sections using deep learning techniques. The proposed system consists of three components: (1) automatically locating tissue regions in the whole slide images (WSIs), (2) splitting located tissue regions into patches and classifying each patch into predefined categories using convolutional neural networks (CNN), and (3) integrating predictions of all patches to form the final diagnosis with a rule-based system. The simple flow chart of the proposed system is shown as below:
<img src="./thyroid_rule_flowchart.png" width="800" height="180" alt="Banner">


### Citation
Please consider cite the paper if this repository facilitates your research.
```
@article{li2020rule,
  title={Rule-based automatic diagnosis of thyroid nodules from intraoperative frozen sections using deep learning},
  author={Li, Yuan and Chen, Pingjun and Li, Zhiyuan and Su, Hai and Yang, Lin and Zhong, Dingrong},
  journal={Artificial Intelligence in Medicine},
  volume={108},
  pages={101918},
  year={2020},
  doi={https://doi.org/10.1016/j.artmed.2020.101918},
  publisher={Elsevier}
}
```
