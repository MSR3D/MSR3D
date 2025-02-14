<h2 align="center">
  <span><img src="asset/logo.jpg" width="4%" style="transform: translate(0,9px)"></span><b> Multi-modal Situated Reasoning in 3D Scenes</b>
</h2>

<div align="center" margin-bottom="6em">
<a target="_blank" rel="external nofollow noopener" href="https://xiongkunlinghu.github.io/">Xiongkun Linghu<sup>✶</sup></a>,
<a target="_blank" rel="external nofollow noopener" href="https://huangjy-pku.github.io/">Jiangyong Huang<sup>✶</sup></a>,
<a target="_blank" rel="external nofollow noopener" href="https://nxsedson.github.io/">Xuesong Niu<sup>✶</sup></a>,
<a target="_blank" rel="external nofollow noopener" href="https://jeasinema.github.io/">Xiaojian Ma</a>,
<a target="_blank" rel="external nofollow noopener" href="https://buzz-beater.github.io/">Baoxiong Jia</a>,
<a target="_blank" rel="external nofollow noopener" href="https://siyuanhuang.com/">Siyuan Huang</a>
</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2409.02389" target="_blank" rel="external nofollow noopener">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://msr3d.github.io/" target="_blank" rel="external nofollow noopener">
    <img src="https://img.shields.io/badge/Page-MSR3D-9cf" alt="Project Page"></a>
    <a href="https://docs.google.com/forms/d/e/1FAIpQLSdzj25kXcthCKtypAI0wQP8j16e9F7ODBroL4SCH_ly8_3rKw/viewform?usp=sf_link" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/Data-MSR3D-blue" alt="Data"></a>
</div>
&nbsp;

<div align="middle">
<img src="asset/MSR3D_teaser_crop.jpeg" width="85%" alt="LEO Teaser">
</div>

### Data Distribution
MSQA includes 251K situated question-answering pairs across 9 distinct question categories, covering complex scenarios within 3D scenes.
<div align="middle">
<img src="asset/data_distribution.png" width="85%" alt="LEO Teaser">
</div>

### Model
MSR3D accepts 3D point cloud, text-image interleaved situation, location, orientation, and question as multi-modal input. It has a stronger situation modeling capability than LEO.
<div align="middle">
<img src="asset/model.png" width="85%" alt="LEO Teaser">
</div>

## 📋 Contents

1. [Introduction](#-introduction)
2. [News](#-news)
3. [TODO List](#📝-todo-list)
4. [Citation](#🔗-citation)
5. [License](#💼-license)
6. [Acknowledgements](#🪧-acknowledgements)

## 📖 Introduction
Situation awareness is essential for understanding and reasoning about 3D scenes in embodied AI agents. However, existing datasets and benchmarks for situated understanding suffer from severe limitations in data modality, scope, diversity, and scale.

To address these limitations, we propose <b>Multi-modal Situated Question Answering (MSQA), a large-scale multi-modal situated reasoning dataset, scalably collected leveraging 3D scene graphs and vision-language models (VLMs) across a diverse range of real-world 3D scenes</b>. MSQA includes <b>251K situated question-answering pairs across 9 distinct question categories, covering complex scenarios within 3D scenes</b>. We introduce a novel interleaved multi-modal input setting in our benchmark to provide both <b>texts, images, and point clouds for situation and question description</b>, aiming to resolve ambiguity in describing situations with single-modality inputs (e.g., texts).


Additionally, we devise the <b>Multi-modal Next-step Navigation (MSNN) benchmark to evaluate models' grounding of actions and transitions between situations</b>. Comprehensive evaluations on reasoning and navigation tasks highlight the limitations of existing vision-language models and underscore the importance of handling multi-modal interleaved inputs and situation modeling. Experiments on data scaling and cross-domain transfer further demonstrate <b>the effectiveness of leveraging MSQA as a pre-training dataset for developing more powerful situated reasoning models</b>, contributing to advancements in 3D scene understanding for embodied AI.

## 🔥 News
- \[2025-2\] We provide the script to visualize the MSQA/MSNN data, including the situaitions.
- \[2024-10\] We released the dataset, which has been structured to facilitate the evaluation of multimodal large language models (MLLMs).
- \[2024-9\] 🎉 Our paper is accepted by NeurIPS 2024 Datasets and Benchmarks Track!
- \[2024-9\] We released the [paper](./asset/MSR3D.pdf) of MSR3D. Please check the [webpage](https://msr3d.github.io/).

## 📝 TODO List

- [x] Test set, with ground truth multi-view images, object locations and attributes
- [x] Train/val set
- [ ] Evaluation code
- [ ] Training code

## 🔗 Citation

If you find our work helpful, please cite:
```bibtex
@article{linghu2024multi,
  title={Multi-modal Situated Reasoning in 3D Scenes},
  author={Linghu, Xiongkun and Huang, Jiangyong and Niu, Xuesong and Ma, Xiaojian and Jia, Baoxiong and Huang, Siyuan},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## 💼 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>

This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 🪧 Acknowledgements
- [LEO](https://embodied-generalist.github.io/): Our baseline model is built upon LEO.
- [SQA3D](https://sqa3d.github.io/): SQA3D is a situated question-answering dataset based on ScanNet.
