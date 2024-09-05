<h2 align="center">
  <span><img src="asset/logo.jpg" width="4%" style="transform: translate(0,9px)"></span><b> Multi-modal Situated Reasoning in 3D Scenes</b>
</h2>

<div align="center" margin-bottom="6em">
<a target="_blank" rel="external nofollow noopener" href="https://github.com/Germany321">Xiongkun Linghu<sup>✶</sup></a>,
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
    <a href="https://drive.google.com/drive/folders/1cRghPKwnYMr6a5DSncFvRB6Ep0T9ywU8" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/Data-MSR3D-blue" alt="Data"></a>
</div>
&nbsp;

<div align="middle">
<img src="asset/MSR3D_teaser_crop.jpeg" width="85%" alt="LEO Teaser">
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
- \[2024-9\] We release the [paper](./asset/MSR3D.pdf) of MSR3D. Please check the [webpage](https://msr3d.github.io/)

## 📝 TODO List

- [x] Test set
- [ ] Train/val set
- [ ] Point cloud and images
- [ ] Evaluation code
- [ ] Training code

## 🔗 Citation

If you find our work helpful, please cite:
```bibtex
@misc{linghu2024multimodalsituatedreasoning3d,
      title={Multi-modal Situated Reasoning in 3D Scenes}, 
      author={Xiongkun Linghu and Jiangyong Huang and Xuesong Niu and Xiaojian Ma and Baoxiong Jia and Siyuan Huang},
      year={2024},
      eprint={2409.02389},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.02389}, 
}
```

## 💼 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>

This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 🪧 Acknowledgements
- [LEO](https://embodied-generalist.github.io/): Our baseline model is built upon LEO.
- [SQA3D](https://sqa3d.github.io/): SQA3D is a situated question-answering dataset based on ScanNet.