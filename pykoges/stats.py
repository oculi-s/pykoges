def normality(
    koges,
    p_threshold=0.05,
    isdisplay=True,
):
    from scipy import stats
    from pykoges.koges import KogesData

    _kg = KogesData.copy(koges)

    res = []
    for x in _kg.x:
        data = _kg.data[x].dropna().astype(float)
        skewness = stats.skew(data)  # 왜도 측정 (<=2)
        if abs(skewness) > 2:
            res.append(x)
        if len(data) >= 3:
            if len(data) < 5000:
                shapiro = stats.shapiro(data)[1]  # 정규성검정 (n<5000)
                # H0 : 변수가 정규분포를 따른다. (p>=th)
                # H1 : 변수가 정규분포를 따르지 않는다. (p<th)
                if shapiro < p_threshold:
                    # res.append(x)
                    continue
            else:
                anderson = stats.anderson(data)  # 정규성검정 (n>=5000)
                # H0 : 변수가 정규분포를 따른다. (검정통계치f>=5%th)
                # H1 : 변수가 정규분포를 따르지 않는다. (검정통계치f<5%th)
                if anderson[0] < anderson[1][2]:
                    # res.append(x)
                    continue
    if isdisplay:
        print("-----------------")
        if len(res):
            print(", ".join(res), "가 정규성을 만족하지 않음")
        else:
            print("모든 변수가 정규성을 만족합니다.")
    # 정규성을 만족하지 않는 변수를 제외합니다.
    # 대부분의 데이터가 정규성을 만족하지 않아 왜도가 2를 넘는 변수만 제외합니다.
    _kg.x = list(set(_kg.x) - set(res))


def homogenity(
    koges,
    p_threshold=0.05,
):
    from scipy import stats
    from pykoges.koges import KogesData

    if not _kg.n_class:
        return

    _kg = KogesData.copy(koges)
    res = []

    for x in _kg.x:
        df_list = []
        for i in range(_kg.n_class):
            df_list.append(_kg.dfs[i][x])
        # H0 : 변수간 분산에 유의미한 차이가 없다. (p>=th)
        # H1 : 변수간 분산에 유의미한 차이가 있다. (p<th)
        bartlett = stats.bartlett(*df_list)[1]
        levene = stats.levene(*df_list)[1]
        if bartlett < p_threshold or levene < p_threshold:
            res.append(x)
    print("-----------------")
    if len(res):
        print(", ".join(res), "가 등분산성을 만족하지 않음")
    else:
        print("모든 변수가 등분산성을 만족합니다.")
    # 등분산성을 만족하지 않는 변수를 제외합니다.
    _kg.x = list(set(_kg.x) - set(res))
    return _kg


def t_test(koges, p_threshold=0.05):
    from scipy import stats
    from pykoges.koges import KogesData
    from pykoges.utils import iscontinuous
    from pykoges.__map import _name_map
    import pandas as pd

    # T-test, 두개 class
    _kg = KogesData.copy(koges)
    new_x, summary = [], []
    for x in _kg.x:
        row = [x]
        if not iscontinuous(_kg.q, x):
            continue
        for i in range(_kg.n_class):
            data = _kg.datas[i][x]
            mean = data.mean()
            lower, upper = stats.t.interval(
                0.95, len(data) - 1, loc=mean, scale=stats.sem(data)
            )
            ci = f"{lower:.2f} - {upper:.2f}"
            row += [f"{mean:.2f}", ci]
        # H0 (귀무가설) : 두 집단의 평균이 같다
        # H1 (대립가설) : 두 집단의 평균이 다르다
        t, p_value = stats.ttest_ind(_kg.datas[0][x], _kg.datas[1][x])
        if p_value < p_threshold:
            new_x.append(x)
        summary.append(row + [p_value])

    _kg.data = pd.concat(_kg.datas.values())
    _kg.data = _kg.data[_kg.x + [_kg.y[0]]]
    print("-----------------------")
    print(f"유효변수 {len(_kg.x)}개")
    print(", ".join(pd.Series(_kg.x).replace(_name_map)))

    _kg.x = sorted(new_x)
    _kg.summary = summary
    return _kg


def anova(koges, p_threshold=0.05):
    from scipy import stats
    from pykoges.koges import KogesData
    from pykoges.utils import iscontinuous
    from pykoges.__map import _name_map
    import pandas as pd

    # ANOVA, 여러 class
    _kg = KogesData.copy(koges)
    new_x, summary = [], []
    for x in _kg.x:
        df_list = []
        row = [x]
        if not iscontinuous(_kg.q, x):
            continue
        for i in range(_kg.n_class):
            data = _kg.datas[i][x]
            df_list.append(data)
            mean = data.mean()
            lower, upper = stats.t.interval(
                1 - p_threshold,
                len(data) - 1,
                loc=mean,
                scale=stats.sem(data),
            )
            ci = f"{lower:.2f} - {upper:.2f}"
            row += [f"{mean:.2f}", ci]
        # H0 : 모든 그룹의 평균이 같다. (p>=th)
        # H1 : 어떤 그룹은 평균이 다르다. (p<th)
        F, p_value = stats.f_oneway(*df_list)
        if p_value < p_threshold:
            new_x.append(x)
        summary.append(row + [p_value])

    _kg.data = pd.concat(_kg.datas.values())
    _kg.data = _kg.data[_kg.x + [_kg.y[0]]]
    print("-----------------------")
    print(f"유효변수 {len(_kg.x)}개")
    print(", ".join(pd.Series(_kg.x).replace(_name_map)))

    _kg.x = sorted(new_x)
    _kg.summary = summary
    return _kg


def split(
    koges,
    n_class=4,
    p_threshold=0.05,
    with_normality=False,
    with_homogenity=False,
):
    from pykoges.koges import KogesData, split_data

    _kg = KogesData.copy(koges)
    if _kg.data.empty:
        print("조건을 만족하는 데이터가 존재하지 않습니다.")
        return _kg
    # 정규성검증 → 데이터 분리 → 등분산성 검정 → T test/ANOVA

    # 1. 정규성 검증
    if with_normality:
        _kg = normality(_kg, p_threshold=p_threshold)
    # 2. 데이터 분리
    _kg = split_data(_kg, n_class=n_class)
    # 3. 등분산성 검정
    if with_homogenity:
        _kg = homogenity(_kg, p_threshold=p_threshold)

    return _kg


def summary(
    koges,
    isdisplay=True,
):
    from pykoges.__map import _name_map
    from IPython.display import display, HTML
    import pandas as pd

    _kg = koges
    # p-value를 기준으로 정렬하고 1e-3보다 작은 경우를 대체합니다.
    summary = pd.DataFrame(_kg.summary)
    summary = summary.sort_values(by=summary.columns[-1])
    summary.iloc[:, -1] = summary.iloc[:, -1].apply(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    )
    index, summary = summary.iloc[:, 0], summary.iloc[:, 1:]
    summary.columns = list(_kg.columns)
    summary.index = list(index.replace(_name_map))
    summary = summary.style.set_table_styles(
        [dict(selector="th", props=[("text-align", "center"), ("white-space", "pre")])]
    )
    _kg.SAVE["statistics"] = summary
    if isdisplay:
        print("-----------------------")
        print(("T test" if _kg.n_class == 2 else "ANOVA") + " 결과")
        display(HTML(summary.to_html()))


def boxplot(
    koges,
    isdisplay=True,
):
    from pykoges.koges import KogesData
    from pykoges.utils import isdiscrete
    from pykoges.__map import _name_map
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    _kg = koges
    col = min(8, len(_kg.x))
    row = (len(_kg.x) + col - 1) // col
    boxplot, ax = plt.subplots(
        nrows=row,
        ncols=col,
        figsize=(col * 1.5, row * 2),
        constrained_layout=True,
        sharey=False,
    )
    for _, x in enumerate(_kg.x):
        df_list = []
        for i in range(_kg.n_class):
            df_list.append(list(_kg.datas[i][x]))
        plt.subplot(row, col, _ + 1)
        sns.boxplot(data=df_list, palette="Set3", showfliers=False)
        plt.xlabel(_name_map.get(x, x))
        if _kg.n_class == 2:
            plt.xticks(range(_kg.n_class), ["(-)", "(+)"])
        elif isdiscrete(_kg.q, _kg.y[0]):
            plt.xticks(range(_kg.n_class))
        else:
            plt.xticks(range(_kg.n_class), [f"Q{i+1}" for i in range(_kg.n_class)])
    plt.suptitle("Boxplot")
    if isdisplay:
        print("-----------------------")
        print("BoxPlot")
        plt.show()

    _kg.SAVE["boxplot"] = boxplot


def __correlation_key(koges):
    from pykoges.utils import isdiscrete, isbinary

    # 연속변수를 기준으로 correlation구함
    for x in [koges.y[0]] + koges.x:
        if x == "age":
            continue
        if not isdiscrete(koges.q, x) and not isbinary(koges.q, x):
            return x
    return koges.y[0]


def correlation(koges):
    from pykoges.utils import iscontinuous, arr_to_df_split
    from pykoges.__map import _name_map
    from IPython.display import display
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    sns.set(font="Malgun Gothic")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    _kg = koges
    if _kg.data.empty:
        print("조건을 만족하는 데이터가 존재하지 않습니다.")
        return
    keys = [x for x in _kg.data if iscontinuous(_kg.q, x)]
    df = _kg.data[keys]
    y = _kg.y[0]
    if iscontinuous(_kg.q, y):
        key = __correlation_key(y)
        corr = (
            pd.DataFrame(df.corr(method="pearson")[key])
            .drop([key], axis=0)
            .sort_values(by=key, ascending=False)
            .T
        )
        dcorr = corr.rename(index=_name_map, columns=_name_map).T.reset_index()
        dcorr = arr_to_df_split(dcorr.values, column=["", _name_map.get(key, key)])
        display(dcorr)
        _kg.SAVE["correlation"] = dcorr
    else:
        fig = plt.figure(figsize=(6, 5))
        corr = df.corr(method="pearson").abs()
        sns.heatmap(corr.rename(index=_name_map, columns=_name_map))
        _kg.SAVE["correlation"] = fig
    _kg.correlation = corr


def scatter(koges):
    from pykoges.utils import iscontinuous
    from pykoges.__map import _name_map
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    _kg = koges
    if _kg.correlation is None or _kg.correlation.empty:
        print("조건을 만족하는 데이터가 존재하지 않습니다.")
        return
    key = __correlation_key(_kg)

    # 가로 8칸에 scatter plot
    keys = [x for x in _kg.correlation if iscontinuous(_kg.q, x) and x != key]
    col = min(8, len(keys) - 1)
    row = (len(keys) + col - 1) // col
    fig, axis = plt.subplots(
        nrows=row,
        ncols=col,
        figsize=(col * 1.8, row * 2),
        constrained_layout=True,
        sharey=True,
    )

    for i, x in enumerate(keys):
        plt.subplot(row, col, i + 1)
        plt.scatter(_kg.data[x], _kg.data[key], alpha=0.1)
        # plt.title(f'{x} - {y_code}')
        plt.xlabel(_name_map.get(x, x))

    fig.supylabel(_name_map.get(key, key))
    fig.suptitle("Scatter plot")
    plt.show()

    _kg.SAVE["scatter"] = fig
