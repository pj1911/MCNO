# Data

Download the benchmark datasets from the [FNO data repository](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) and place them in this directory.

## Available Datasets

### KdV Equation

- **File**: `kdv_train_test.mat`
- **Contents**: Input-output pairs for the Korteweg–de Vries equation
  - `input`: Initial conditions (shape: `[N, 8192]`)
  - `output`: Corresponding solutions (shape: `[N, 8192]`)
- **Data generation**: [mwt-operator/Data/kDV](https://github.com/gaurav71531/mwt-operator/tree/master/Data/kDV)

### Burgers Equation

- **Files**: `Burgers_R10.zip`, `Burgers_v100.zip`, or `Burgers_v1000.zip`
- **Contents**: Input-output pairs for the viscous Burgers equation at different resolutions

## Directory Structure

After downloading, this directory should contain:

```
data/
├── README.md
├── kdv_train_test.mat
└── burgers_data_R10.mat   (optional)
```
