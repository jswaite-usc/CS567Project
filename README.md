# CSCI-567: USC

# Project Title: Explorations of Regularizations on the Various Stages of Convolutional Neural Networks (CNNs)

## Abstract

Convolutional Neural Networks (CNNs) have showcased remarkable performance in image classification tasks. However, their susceptibility to overfitting poses a challenge in generalizing to unseen data. To address this issue, various regularization techniques have been proposed. This project presents a survey and experimental investigation of regularization methods for CNNs, categorizing them into input, internal, and label techniques. The research explores the impact of these methods on training and test errors, both individually and in combination. Using the CIFAR-100 and CIFAR-10 datasets, we evaluate techniques such as Cutout, Random Erasing, Mixup, Dropout, Max Dropout, Drop Block, Label Smoothing, and TSLA. This study aims to provide insights into effective regularization strategies and pave the way for future advancements in improving CNN generalization.

## Dataset Repository Link
https://www.cs.toronto.edu/~kriz/cifar.html

## Libraries Used

- **matplotlib** (Version 3.7.1)
- **opencv-python** (Version 4.8.0.76)
- **pandas** (Version 2.0.3)
- **scikit-learn** (Version 1.2.2)
- **seaborn** (Version 0.13.1)
- **sklearn-pandas** (Version 2.2.0)
- **tensorboard** (Version 2.15.2)
- **tensorflow** (Version 2.15.0)
- **tf_keras** (Version 2.15.1)
- **torch** (Version 2.2.1+cu121)

## Contributors

- Joonyoung Bae
- Muhammad Adil Fayyaz
- Matthew Salaway
- Jacob Waite

## Instructions for Replication

To replicate the experiments conducted in this project, follow these steps:

1. Install the required libraries listed above using pip or any other package manager.

```bash
pip install matplotlib==3.7.1 opencv-python==4.8.0.76 pandas==2.0.3 scikit-learn==1.2.2 seaborn==0.13.1 sklearn-pandas==2.2.0 tensorboard==2.15.2 tensorflow==2.15.0 tf_keras==2.15.1 torch==2.2.1+cu121
```

2. Clone the project repository from GitHub.

```bash
git clone <repository_url>
```

3. Navigate to the project directory.

```bash
cd <project_directory>
```

4. Run the provided scripts or notebooks to reproduce the experiments and analysis conducted in the project.

## License

This project is licensed under the [MIT License](LICENSE).
