# Introduction

This repository contains the script to train text classifier.

# Setup

To install the dependencies, you can simply follow this steps.

To create and activate the virtual environment, follow these steps:

**Using `conda`**

```bash
# To create a virtual enviornment:
$ conda create --name env

# Activate the virtual environment:
$ conda activate env

# To deactivate (when you're done):
(env)$ conda deactivate
```

**Using `virtualenv`**

```bash
# To create a virtual enviornment:
$ virtualenv env

# Activate the virtual environment:
$ source env/bin/activate
# OR
$ env\Scripts\activate

# To deactivate (when you're done):
(env)$ deactivate
```

To install the requirements using `pip`, once the virtual environment is active:
```bash
(env)$ pip install -r requirements.txt
```
# Data Preparation

The training excel file is stored in the [data folder](https://github.com/chengjiangg/BT4103-Group3/tree/main/training_script/data) and have the following column names:

|   text    |   entity  |  emotion  |   stance  |
|-----------|-----------|-----------|-----------|

# Training

The model can be trained end-to-end with the following code:

```bash
python main.py --excel_filename en_dataset.xlsx --sheet_name Sheet1 --classifer_type en
```

**NOTE**
- `log_filename`, and `saved_model_name` should be specified to avoid overwriting the original file. 
- `classifier_type` model language to initialized based on the [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes). Current supported language code **[en, zh]**.