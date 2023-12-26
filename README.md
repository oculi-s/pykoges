# PyKoges 데이터 분석 패키지

PyKoges는 Python 기반의 코호트 데이터 분석 도구로, 특히 [Koges](https://nih.go.kr/ko/main/contents.do?menuNo=300566) (한국인유전체역학조사사업) 데이터를 처리, 분석, 모델링하는데 특화되어 있습니다. 다음은 PyKoges 패키지를 사용하여 데이터를 읽고, 전처리하며, 분석 및 예측 모델을 구축하는 예시입니다.

:exclamation: 데이터가 포함되지 않은 분석 도구입니다. :exclamation:

```
pip install pykoges
```

## 시작하기 전에

PyKoges 패키지를 사용하기 전에 필요한 모듈을 임포트합니다.

```python
from pykoges import codingbook, koges, stats, model
```

## 1. 코딩북 읽기

데이터 분석을 위해 먼저 코딩북을 읽어와야 합니다.

```python
q = codingbook.read(folder_name='./data_fixed/')
# q.summary() # 코딩북의 요약 정보를 확인하고 싶다면 이 코드를 실행합니다.
```

## 2. 변수 설정

분석에 사용할 변수를 설정합니다.

```python
x_list = {
    # 여기에 입력 변수를 설정합니다...
}

y_list = {
    # 여기에 출력 변수를 설정합니다...
}

patientinfo_list = {
    # 환자를 구분할 수 있는 정보를 넣습니다...
}

# 변수 설정을 PyKoges에 적용합니다.
var = koges.Variables(
    q=q,
    x_list=x_list,
    y_list=y_list,
    patientinfo_list=patientinfo_list,
)
var.summary(
    display_datainfo=True,
    display_userinfo=True,
)
```

## 3. 데이터 읽기

설정된 변수를 바탕으로 데이터를 읽어옵니다.

```python
kg = koges.read(
    variables=var,
    folder_name='./data_fixed/',
    filter_patient=True,
)
```

## 4. 데이터 전처리

데이터를 전처리합니다. 여기에서는 몇 가지 전처리 옵션을 선택할 수 있습니다.

```python
custom_functions = [
    # 여기에 원하는 커스텀 함수를 설정할 수 있습니다.
    #  ([x, y], f)의 형태로 설정합니다.
]

kg_converted = koges.convert(
    koges=kg,
    muscle_weight_ratio=False, # muscle/weight
    muscle_height_ratio=False, # muscle/height
    muscle_bmi_ratio=False, # muscle/bmi
    waist_hip_ratio=False, # waist/hip
    fev_fvc_ratio=False, # fev1/fvc
    grip_of_grwhich=True, # 주 사용손 악력
    weight_height_bmi=False, # bmi
    custom_functions=custom_functions,
)
```

## 5. 결측치 제거

결측치 처리 및 이상치 제거 등을 진행합니다.

```python
kg_dropped = koges.drop(
    koges=kg_converted,
    drop_threshold=0.3, # drop_threshold 이상이 결측치인 변수 제거
    filter_alpha=3, # 3SD를 벗어나느 데이터 제거
    data_impute=False, # KNN알고리즘으로 결측치 채우기
    display_result=True,
    display_count=True,
)
```

## 6. 통계 분석

### 데이터 분리

```python
kg_splitted = stats.split(
    koges=kg_dropped,
    n_class=4, # 그룹 개수
    p_threshold=0.05,
    with_normality=False,
    with_homogenity=False,
    isdisplay=True
)
```

### 통계 분석 수행

```python
if kg_splitted.n_class == 2:
    kg_stats = stats.t_test(koges=kg_splitted, p_threshold=.05)
else:
    kg_stats = stats.anova(koges=kg_splitted, p_threshold=.05)
```

### 분석 결과 요약

```python
stats.summary(koges=kg_stats, isdisplay=True)
```

<center>

![](/example/4.statistics.png)

</center>

### 분석 결과 boxplot

```python
stats.boxplot(koges=kg_stats, isdisplay=True)
```

<center>

![](/example/5.boxplot.png)

</center>

### 상관관계, 산점도 분석

```python
stats.correlation(koges=kg_stats, isdisplay=True)
stats.scatter(koges=kg_stats, isdisplay=True)
```
<center>

![](/example/7.scatter.png)

</center>

## 7. 머신러닝

머신러닝 모델을 선택하여 훈련시키고 예측 진행을 합니다.  
여/부는 로지스틱, 이산변수는 softmax, 연속변수는 선형 회귀를 진행합니다.

```python
ml = model(
    koges=kg_stats,
    scalers=["minmax", "robust", "standard", "maxabs"],
)

if kg_stats.type == "binary":
    ml.logistic(isdisplay=True)
elif kg_stats.type == "discrete":
    ml.softmax(
        display_roc_curve=True,
        display_confusion_matrix=True,
    )
elif kg_stats.type == "continuous":
    ml.linear(isdisplay=True)
    # regression의 결과가 안좋은 경우 quantile classification도 진행합니다.
    if ml.r2 < 0.8:
        ml.softmax(
            display_roc_curve=True,
            display_confusion_matrix=True,
        )
```

예시이미지

<center> 

![](example/1.LinearRegression.png)

</center>

## 8. 결과 저장

모든 작업의 결과물과 모델을 `/result` 폴더에 저장합니다.

```python
ml.equation(isdisplay=True)
kg_stats.save(isdisplay=True)
```

이 문서는 PyKoges 패키지의 핵심 기능과 사용 방법을 제공하는 가이드라인을 제시합니다. 실제 데이터 분석 시 이 문서를 참고하여 PyKoges를 활용해보세요.