# Paint2Code

Welcome to Paint2Code, a robust tool designed to transform your hand-drawn sketches into functional HTML code. This innovative project leverages advanced image recognition and machine learning algorithms to interpret drawings and convert them into clean, structured HTML elements.

## Features

- **Image to HTML Conversion**: Upload your sketch and receive HTML code.
- **Support for Multiple HTML Elements**: Detects various shapes and interprets them as different HTML elements.
- **Easy to Use Interface**: User-friendly interface designed for both beginners and advanced users.
- **Multiple Encoder Models**: Utilize various encoder models to enhance accuracy and flexibility in interpreting sketches.
- **Support for Multiple HTML Code Styles**: Choose from different HTML coding styles to match your project requirements.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```
- Python 3.8 or higher
- matplotlib==3.8.0
- nltk==3.8.1
- numpy==1.26.4
- Pillow==10.3.0
- streamlit==1.33.0
- torch==2.2.1
- torchvision==0.17.1
- tqdm==4.65.0
```

### Installing

A step-by-step series of examples that tell you how to get a development environment running:

1. Clone the repo:
   ```bash
   git clone https://github.com/nico1008/paint2code
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the original Paint2Code dataset, please visit [pain2code Dataset](https://www.kaggle.com/datasets/nico1008/paint2code) to download it.

### Model Weights
- CustomCNN: [Download here](https://www.dropbox.com/scl/fi/27xuj0qi6xrfbbdkyryva/ED-epoch-85-loss-0.01651.rar?rlkey=e20s38l7y02w6oativr5fuiw0&st=63xrfz6l&dl=0)
- MobileNetV3: [Download here](https://www.dropbox.com/scl/fi/nfwdxlz07qo9fot0vhg42/ED-epoch-73-loss-0.03660.rar?rlkey=1trsupm6jdsm9fqsq44mljy3w&st=andynm0f&dl=0)
- ResNet18: [Download here](https://www.dropbox.com/scl/fi/sk40o33fp7zfrk9lnvzgo/ED-epoch-105-loss-0.03323.rar?rlkey=5zku77indq6tolg4kyix276k8&st=1ppwhxlv&dl=0)

### Data Placement
Place your training data in the `.data/all_data` folder.

### Methods of Use
There are two methods to use this project: via Jupyter notebooks or Python scripts.
I suggest using the Jupyter notebooks for better data visualisation.

#### Method 1: Using Jupyter Notebooks

1. **Data Preparation**: Open `prepareData.ipynb` and run all cells.
2. **Model Training**: Open the appropriate notebook for desired model and run all cells:
   - `trainCustomCNN.ipynb` for CustomCNN
   - `trainMobileNet.ipynb` for MobileNetV3
   - `trainResNet18.ipynb` for ResNet18
3. **Model Evaluation**: Open the corresponding evaluation notebook and run all cells:
   - `evalCustomCNN.ipynb` for CustomCNN
   - `evalMobileNet.ipynb` for MobileNetV3
   - `evalResNet18.ipynb` for ResNet18

#### Method 2: Using Python Scripts

1. **Data Preparation**: Run `python prepareData.py`.
2. **Model Training**: Run `python train.py`.
3. **Model Evaluation**: Run `python eval.py`.

## Acknowledgments

I would like to thank the following resources, communities and people for their invaluable contributions and support:

- The [OpenCV](https://opencv.org/) team for their robust computer vision library.
- The [PyTorch](https://pytorch.org/) community for their deep learning framework.
- [KKopilka](https://github.com/KKopilka) for major moral support and love. 
- [Tony Beltramelli](https://github.com/tonybeltramelli/pix2code) project draws significant inspiration from Tony Beltramelli's pioneering Pix2Code modell.

## Contact Me

If you have any questions, suggestions, or need further assistance, please feel free to reach out:

- Telegram: [@nico_1008k](https://t.me/nico_1008k)
- GitHub: [nico1008](https://github.com/nico1008)

---

I hope you find Paint2Code useful! If you have any questions or feedback, please feel free to reach out.