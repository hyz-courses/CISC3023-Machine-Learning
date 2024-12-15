# CISC3023 Machine Learning Final Project
- Title: Wound Area Location In Animal Model Images
- Groupmates: Huang Yanzhen DC126732, Yang Zhihan DC127992

## Files

Original file: [main.ipynb](main.ipynb)

Code preview:
- [HTML Version](code_preview.html) 
- [PDF Version](code_preview.pdf)

## Models
*Grid Search Object Lists*
- [`RandomForestRegressor_nest-maxd_17328034630916922.sav`](./model/RandomForestRegressor_nest-maxd_17328034630916922.sav) 
- [`RandomForestRegressor_mins-minl_173285867054318.sav`](./model/RandomForestRegressor_mins-minl_173285867054318.sav) 
- [`SVR_krnl-C_1732885935665518.sav`](./models/SVR_krnl-C_1732885935665518.sav)
- [`SVR_epsl-gamm_1732886643413752.sav`](./models/SVR_epsl-gamm_1732886643413752.sav)

*Best Models*
- [`best_model_rfr_1733156550837645.sav`](./models/best_model_rfr_1733156550837645.sav)
- [`best_model_svr_1733156169715915.sav`](./models/best_model_svr_1733156169715915.sav)

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


2. To view and run code, open jupyter notebook and view [main.ipynb](./main.ipynb). You could also view the PDF version of the code at [code_preview.pdf](./code_preview.pdf), or the HTML version at [code_preview.html](./code_preview.html).


> P.S.: The trained models are also available from Google Drive:
https://drive.google.com/drive/folders/1v_wMipy_cq6g4gG5CH8lL3B7zYcoTo04?usp=sharing
All the models need to be moved to `.sav` files into the `models` folder, which sould be at the same level as `main.ipynb`.
