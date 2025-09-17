# Acknowledgements

We express our gratitude to the researchers, developers, and communities
whose foundational work and contributions made GFPGAN possible.

## Research Foundations

### Original Research

This project builds upon the foundational research and implementations from:

**GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior**
*Xintao Wang, Yu Li, Honglun Zhang, Ying Shan*
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021
[Paper](https://arxiv.org/abs/2101.04061) | [Original Implementation](https://github.com/TencentARC/GFPGAN)

We acknowledge and thank the original authors for their groundbreaking
research in generative facial priors and their open-source contribution
that enabled this project. This project has been completely unforked and
operates independently while preserving attribution to the original
research.

## Core Dependencies

### Essential Libraries

**BasicSR** - Super-Resolution Framework
*Xintao Wang and contributors*
[GitHub](https://github.com/xinntao/BasicSR)
Provides the fundamental training and inference framework for super-resolution models.

**FaceXLib** - Face Detection and Analysis
*Xintao Wang and contributors*
[GitHub](https://github.com/xinntao/facexlib)
Essential for face detection, alignment, and facial feature analysis.

### Model Architectures

**StyleGAN2** - Generative Adversarial Networks
*Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko
Lehtinen, Timo Aila*
Provides the core generative architecture for high-quality face synthesis.

**Real-ESRGAN** - Real-World Image Super-Resolution
*Xintao Wang and contributors*
Background enhancement and general image upscaling capabilities.

## Technical Infrastructure

### Deep Learning Frameworks

**PyTorch** - Machine Learning Framework
*Facebook AI Research and contributors*
The foundation for all model training and inference operations.

**OpenCV** - Computer Vision Library
*Intel Corporation and contributors*
Essential for image processing, manipulation, and I/O operations.

### Web and API Framework

**Gradio** - Machine Learning Web Interfaces
*Abubakar Abid and the Gradio team*
Enables the intuitive web interface for interactive face restoration.

**FastAPI** - Modern Web Framework
*Sebastián Ramírez and contributors*
Powers the REST API for programmatic access to restoration capabilities.

## Community and Development

### Development Tools

**MkDocs Material** - Documentation Framework
*Martin Donath and contributors*
Provides the beautiful and functional documentation site.

**Ruff** - Python Linting and Formatting
*Charlie Marsh and contributors*
Ensures code quality and consistency throughout the project.

**Pre-commit** - Git Hook Framework
*Anthony Sottile and contributors*
Maintains code quality standards and automated checks.

### Testing and Quality Assurance

**pytest** - Testing Framework
*Holger Krekel and contributors*
Comprehensive testing infrastructure for reliability and quality.

**GitHub Actions** - CI/CD Platform
*GitHub and contributors*
Automated testing, building, and deployment workflows.

## Data and Evaluation

### Evaluation Datasets

**CelebA-HQ** - High-Quality Celebrity Faces
*Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen*
High-resolution celebrity face dataset for evaluation and benchmarking.

**FFHQ** - Flickr-Faces-HQ Dataset
*Tero Karras, Samuli Laine, Timo Aila*
Diverse, high-quality face dataset for comprehensive evaluation.

### Metrics and Evaluation

**LPIPS** - Learned Perceptual Image Patch Similarity
*Richard Zhang, Phillip Isola, Alexei A. Efros*
Perceptual similarity measurement for image quality assessment.

**ArcFace** - Additive Angular Margin Loss
*Jiankang Deng, Jia Guo, Stefanos Zafeiriou*
Identity preservation evaluation through facial recognition embeddings.

## Inspiration and Related Work

### Face Restoration Research

**DFDNet** - Deep Face Dictionary Network
*Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo, Lei Zhang*
Pioneering work in dictionary-based face restoration.

**CodeFormer** - Learning to Restore Face Images
*Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, Chen Change Loy*
Advanced transformer-based approach to face restoration.

**RestoreFormer** - High-Quality Blind Face Restoration
*Zhouxia Wang, Jiawei Zhang, Runjian Chen, Wenping Wang, Ping Luo*
State-of-the-art transformer architecture for face enhancement.

## Community Contributions

### Contributors

We thank all community members who have contributed to this project through:

- Code contributions and bug fixes
- Documentation improvements
- Issue reporting and testing
- Feature suggestions and feedback
- Community support and discussions

### Special Recognition

- **Beta testers**: Early adopters who provided valuable feedback
- **Documentation reviewers**: Contributors who improved clarity and accuracy
- **Accessibility advocates**: Those who helped improve usability for all users
- **Security researchers**: Responsible disclosure of security considerations

## Institutional Support

### Research Community

- **Computer Vision research community**: For advancing the field of face restoration
- **Open-source community**: For fostering collaboration and knowledge sharing
- **AI ethics researchers**: For guidance on responsible AI development

### Standards and Guidelines

**Model Cards for Model Reporting**
*Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy
Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit
Gebru*
Framework for transparent model documentation.

**Datasheets for Datasets**
*Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman
Vaughan, Hanna Wallach, Hal Daumé III, Kate Crawford*
Guidelines for comprehensive dataset documentation.

## License and Legal

### Open Source Licenses

This project is made possible by the generous open-source licenses of our dependencies:

- **Apache License 2.0**: Core project license ensuring freedom to use,
  modify, and distribute
- **MIT License**: Many utility libraries and tools
- **BSD Licenses**: Scientific computing and computer vision libraries
- **Creative Commons**: Documentation and educational content

### Patent Considerations

We acknowledge that some techniques used in this project may be covered by
patents. Users should ensure compliance with applicable patent laws in
their jurisdiction.

## Disclaimer

While we strive to acknowledge all contributors and influences, this list
may not be exhaustive. If you believe your work should be acknowledged
here, please contact us at
[acknowledgements@gfpgan.ai](mailto:acknowledgements@gfpgan.ai).

The inclusion of any work, dataset, or contribution in these acknowledgements
does not imply endorsement of this project by the original authors or
institutions.

---

**Contributing to acknowledgements**:
If you've contributed to this project or believe your work should be acknowledged,
please [open an issue](https://github.com/IAmJonoBo/Restoria/issues) or
contact us directly.
Alternatively, you can reach the maintainers via email if GitHub is not an option.

**Citation**:
When citing this project, please also consider citing the foundational research and
key dependencies that make this work possible.
