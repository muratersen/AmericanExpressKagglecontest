# AmericanExpressKagglecontest
This is my python code for kaggle contest:https://www.kaggle.com/competitions/amex-default-prediction
I used a simple LGBM algorithm in the code. Amex score of the submission is:0.80467
Data folder contains testa.parquet,traina.parquet and train_labels.csv which are test features train features and train labels respectively.
Features are in .parquet form since parquet uses less memory.Detailed explanation can be found here:https://www.kaggle.com/competitions/amex-default-prediction/discussion/327138
You may use modify this code also if you need csv format of the features dont hesitate to contact me.
Featuresincsv folder contains features in csv format
For running training.py put testa.parquet,traina.parquet and training.py in the same folder also modify the line 22 according to your current working directory.
