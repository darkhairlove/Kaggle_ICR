# [Kaggle] ICR - Identifying Age-Related Conditions

## About
세 가지 연령 관련 질환과 이에 관련된 50개 이상의 익명화된 건강 특성으로 구성되어 있는 데이터가 있습니다.  
이 대회의 목표는 피험자가 이러한 질환 중 하나로 진단되었는지 여부를 예측하는 binary classification 문제입니다.

## Project structure
```
Folder/
|- EDA/             # EDA (ipynb)
|- Hyperparameter/  # AutoML (ipynb)
|- MODEL/           # final model (ipynb)
|- Reference/       # 내용 정리본 (pdf, markdown)
```


## Skill
- KNNImputer
- Standardscaler
- Oversampling
- StratifiedKFold
- CV Stacking (Ensemble)

## Dataset
### Data Source
  [Train Test Greeks](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data) 🔗
### Data Info.
- `train.csv`

    - `Id` 각 관측값에 대한 고유 식별자입니다.

    - `AB-GL` 익명화된 56개의 건강 특성. categorical인 `EJ`를 제외하고 모두 숫자입니다.

    - `class` binary target : `1`은 피험자가 세 가지 조건 중 하나로 진단받았음을 나타내고, `0`은 진단받지 않았음을 나타냅니다.

- `test.csv` : 피험자가 두 `class` 각각에 속할 확률을 예측하는 것

- `greeks.csv` - 훈련 집합에만 사용할 수 있는 보조 metadata

    - `Alpha` : 연령 관련 조건이 있는 경우 해당 유형을 식별합니다.
        - `A` : 연령 관련 조건이 없습니다. 클래스 `0`에 해당합니다.
        - `B, D, G` : 세 가지 연령 관련 조건. 클래스 `1`에 해당합니다.

    - `Beta`, `Gamma`, `Delta` : 세 가지 실험 특성입니다.
    - `Epsilon` : 이 피험자에 대한 데이터가 수집된 날짜입니다. 테스트 세트의 모든 데이터는 훈련 세트가 수집된 후에 수집되었습니다.

- `sample_submission.csv`- 올바른 형식의 샘플 제출 파일


## Feature Engineering & Preprocessing

### Preprocessing features

- `KNNImputer` : 결측치 보간

### Feature Engineering
- `Label Encoding` : ‘EJ’ column
    - 'A': 0, 'B': 1
- `StandardScaler` : sklearn으로 Data Scaling
- `Oversampling` : `greeks.Alpha`를 기준으로 SMOTE으로 617 &rarr;  2036 Oversampling
- `PCA, VIF` : Column을 줄이기 위해 사용했지만, balancedlogloss가 좋아지지 않아 사용하지 않음.

## Modeling
### Model

- XGBClassifier    
- $\color{red}{\textsf{LGBMClassifier}}$
- $\color{red}{\textsf{CatBoostClassifier}}$
- $\color{red}{\textsf{HistGradientBoostingClassifier}}$
- $\color{red}{\textsf{RandomForestClassifier}}$

### Hyperparameter
- [Optuna](https://github.com/darkhairlove/Kaggle_ICR/blob/main/Hyperparameter/Optuna_Automl.ipynb) 🔗
- [Flaml](https://github.com/darkhairlove/Kaggle_ICR/blob/main/Hyperparameter/Flaml_Automl.ipynb) 🔗
  
### Evaluation
-  `balanced logarithmic loss` :  [대회 평가 지표에 대한 설명](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview/evaluation) 🔗


### Cross Validation
  - KFold
  - $\color{red}{\textsf{StratifiedKFold}}$
  - MultilabelStratifiedKFold

### Ensemble
  - Stacking
  - $\color{red}{\textsf{CV Stackaing}}$
## Result
- Leaderboard Score : 0.46
- Cross Validation Score : 0.338992


## Member
- [김예진](https://github.com/hanishereandnow)
- [강덕훈](https://github.com/Deok-Hun)
- [박수진](https://github.com/darkhairlove)
