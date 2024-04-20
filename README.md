# diabetes

Ridge regression for predicting diabetes

## First and foremost
Clone the repository using git.

## Installation instructions
1. Install python>=3.10 using anaconda preferably.
2. Create a virtual environment with name `diabetes`. Run `conda create -n diabetes python=3.11`  
3. `conda activate diabetes`
4. `pip install -r requirements.txt`
5. `pip install -e .`

After running above commands, you are all set up to run the code.

## Running the code
Always change your working directory to `src/` before running any code.

## Grid Search
To perform grid search, run `python main.py --hypergradient=False --seed=42 --lmbd_grid="0.0,0.1,1.0,10.0,100.0"`  
An image with validation loss vs lambda will be generated in the `results/` folder with the name `grid_search.png`

## Hypergradient Hyperparameter Search
To perform hypergradient based hyperparameter search, run `python main.py --hypergradient=True --seed=42 --lr=1e-3 --num_epochs=1000`
Two images with (1) validation loss vs hypergradient iterations and (2) lambda vs hypergradient iterations are generated with the names `hypergradient_valloss.png` and `hypergradient_lmbd.png` respectively.


## Visualization in `wandb`
Each time you run the code, the URL to the corresponding run in Wandb will appear on the terminal.
