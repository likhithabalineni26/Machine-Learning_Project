CROP YIELD PREDICTION USING RIDGE, LASSO, AND ELASTIC NET REGRESSION

Project type:
Machine Learning regression project implemented in Python.

How to run in VS Code:
1. Open this folder in VS Code.
2. Open Terminal -> New Terminal.
3. Create a virtual environment (optional but recommended):
   python -m venv venv
4. Activate it on Windows:
   venv\Scripts\activate
5. Install packages:
   pip install -r requirements.txt
6. Run the project:
   python main.py

The program:
- Loads the crop dataset from data/crop_yield.csv
- Understands the dataset
- Preprocesses numerical and categorical features
- Performs exploratory data analysis
- Creates useful features
- Trains Ridge, Lasso, and Elastic Net Regression
- Evaluates MAE, MSE, RMSE, and R2
- Saves the best model in models/
- Saves evaluation results in outputs/
- Saves plots in plots/

Dataset note:
The included CSV is a small sample dataset created for testing the project.
Replace data/crop_yield.csv with a larger real agricultural dataset using the same column names,
or update module3_preprocessing.py if your real dataset has different columns.
