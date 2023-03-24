# BT4103 - Group3

<!-- Project description -->
This repository contains code to build a Streamlit web app that serves Multi-Lingual Text Classifier.

## Prerequisities

Before you begin, ensure you have met the following requirements:

* You have a _Windows/Linux/Mac_ machine running [Python 3.8](https://www.python.org/).
* You have installed the latest versions of [`pip`](https://pip.pypa.io/en/stable/installation/) and [`virtualenv`](https://virtualenv.pypa.io/en/latest/installation.html) or `conda` ([Anaconda](https://www.anaconda.com/distribution/)).


## Setup

To install the dependencies, you can simply follow this steps.

Clone the project repository:
```bash
git clone https://github.com/chengjiangg/BT4103-Group3.git
cd BT4103-Group3/user_interface
```

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

#### Running the script

Finally, if you want to run the main script:
```bash
(env)$ streamlit run MyApp.py
```
