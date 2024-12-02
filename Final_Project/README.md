# CISC3023 Machine Learning Final Project
- Title: Wound Detection
- Groupmates: Huang Yanzhen DC126732, Yang Zhihan DC127992

## Github
The source code and other assets of this final project is stored on GitHub at: https://github.com/hyz-courses/CISC3023-Machine-Learning/tree/master/Final_Project.

This is the sub-folder of the repository for this course: https://github.com/hyz-courses/CISC3023-Machine-Learning.git

## Steps to run code
1. Install all the dependencies and make sure it is in the correct conda virtual environment.
If you use windows, run:
```sh
<LOCATION_OF_YOUR_VIRTUAL_ENV>/<NAME_OF_YOUR_VENV>/Scripts/pip.exe install -r requirements.txt
```
If you use mac, run:
```sh
<LOCATION_OF_YOUR_VIRTUAL_ENV>/<NAME_OF_YOUR_VENV>/bin/pip install -r requirements.txt
```
If you are used to use the base environment and you are sure that all the packages are satisfied in the base environment, you can simply run:
```sh
pip install -r requirements.txt
```

All the required packages are listed in [requirements.txt](./requirements.txt).

2. If you are opening from Moodle's submission zip, you can ignore this step. However, if you are opening from github, you need to download the models from Google Drive since they are too large to be uploaded to github:
https://drive.google.com/drive/folders/1v_wMipy_cq6g4gG5CH8lL3B7zYcoTo04?usp=sharing

Moreover, you need to move all the `.sav` files into the `models` folder, which sould be at the same level as `main.ipynb`.

3. To view and run code, open jupyter notebook and view [main.ipynb](./main.ipynb). You could also view the PDF version of the code at [code_preview.pdf](./code_preview.pdf).