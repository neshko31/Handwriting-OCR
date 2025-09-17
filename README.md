# NANS-Project

An Optical Character Recognition (OCR) model that is used for recognizing and extracting text from images containing characters in both Cyrillic and Latin alphabets. 
Built as a project for NANS subject on Faculty of Technical Sciences in Novi Sad.

## Features

- **Dual-script support**: Recognizes both Cyrillic and Latin characters
- **Designed algorithm**: Row extraction algorithm
- **Flexible input**: Supports various image formats
- **Preprocessing**: Built-in image enhancement for better recognition

## Getting Started

### Prerequisites

- **Python 3.11.0** (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/neshko31/NANS-Project.git
cd NANS-Project
```

2. Install dependencies:

   **Mac/Linux:**
   ```bash
   python3.11  -m  venv  .venv
   source  .venv/bin/activate
   pip install -r requirements/requirements.txt
   ```
   
   **Windows:**
   ```bash
   py  -3.11  -m  venv  .venv
   .venv\Scripts\activate
   pip install -r requirements/requirements_win.txt
   ```

3. Download dataset:
Dataset used: **[OpenWrite Dataset - Version 2](https://www.kaggle.com/datasets/nenadlukic/openwrite-dataset/versions/2)**
After downloading the dataset, folder structure should be as following:
```bash
NANS-Project
├───.venv # Kreira se kad napravite virtuelno okruženje, korak 4.
├───data
│   ├───comnist-train-data
│   ├───cyrillicmnist-train-data
│   ├───emnist-train-data
│   ├───test
│   ├───train_processed
│   ├───train_processed_comnist
│   ├───train_processed_cyrillicmnist
│   ├───train_processed_emnist
│   └───val
├───model
└───requirements
```

## Supported Languages

- **Latin scripts**: English
- **Cyrillic and Latin scripts**: Serbian

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

- **[neshko31](https://github.com/neshko31)**
- **[MarijaT7](https://github.com/MarijaT7)**

## Literature and used datasets
- **[Soft Computing 2024](https://github.com/ftn-ai-lab/sc-2024)**
- **[Soft Computing 2024 Solutions](https://github.com/neshko31/sc-2024-resenja)**
- **[OpenWrite Dataset - Version 2](https://www.kaggle.com/datasets/nenadlukic/openwrite-dataset/versions/2)**