# CISC3023 Machine Learning Final Project
- Title: Wound Area Location In Animal Model Images
- Groupmates: Huang Yanzhen DC126732, Yang Zhihan DC127992

## Files

**Original ipynb file:** [main.ipynb](main.ipynb)

**Code preview:**
- [HTML Version](code_preview.html) 
- [PDF Version](code_preview.pdf)

**Models:**

*Grid Search Object Lists*

> This is the results of all the grid search. To save space, they are placed in the google drive https://drive.google.com/drive/folders/1v_wMipy_cq6g4gG5CH8lL3B7zYcoTo04?usp=sharing. Please move them to the `/models` folder in the root directory.

- `/models/RandomForestRegressor_nest-maxd_17328034630916922.sav`
- `/models/RandomForestRegressor_mins-minl_173285867054318.sav `
- `/models/SVR_krnl-C_1732885935665518.sav`
- `/models/SVR_epsl-gamm_1732886643413752.sav`

*Best Models*
> They are stored under the `/models` folder in the root directory.
- `/models/best_model_rfr_1733156550837645.sav`
- `/models/best_model_svr_1733156169715915.sav`

## Steps to run code
**Step 1.** Install all the dependencies and make sure it is in the correct conda virtual environment.

*Step 1.1* Ignore this step if you have your own virtual environment. If you don't, however, create a conda environment. Open anaconda prompt, run:
```sh
conda create --prefix <LOCATION_OF_YOUR_VIRTUAL_ENV>/<NAME_OF_YOUR_VENV> python=3.12
```

*Step 1.2* Activate your virtual environment.

```console
conda activate <LOCATION_OF_YOUR_VIRTUAL_ENV>/<NAME_OF_YOUR_VENV>
```

*Step 1.3* Install all the required packages.

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

---

**Step 2.** To view and run code, open jupyter notebook and view [main.ipynb](./main.ipynb). 

*Step 2.1* Activate your virtual environment.

```console
conda activate <LOCATION_OF_YOUR_VIRTUAL_ENV>/<NAME_OF_YOUR_VENV>
```

*Step 2.2* Run this command.
```console
jupyter notebook
```

> To make sure `jupyter notebook` can work, please make sure that you have installed the python package `notebook` and `ipykernel`.

> You could also view the PDF version of the code at [code_preview.pdf](./code_preview.pdf), or the HTML version at [code_preview.html](./code_preview.html).



