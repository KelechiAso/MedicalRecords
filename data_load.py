import pandas as pd

clients = pd.read_csv('data\patients.csv')
enc = pd.read_csv('data\encounter.csv')
dx = pd.read_csv('data\diagnosis.csv')
vx = pd.read_csv('data\vitals.csv')
meds = pd.read_csv('data\Medications.csv')