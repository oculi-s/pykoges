# 변수가 이진변수인지 확인
def isbinary(q, code):
    from pykoges.__map import _type_map

    if code in _type_map:
        return _type_map[code] == 0
    answer = next(iter(q.has_code(code).answer), None)
    if not answer:
        return False
    # O, X, 무응답
    keys = answer.keys()
    return 0 < len(set(keys) - {"0", "9"}) <= 3


# 변수가 이산변수인지 확인
def isdiscrete(q, code):
    from pykoges.__map import _type_map

    if code in _type_map:
        return _type_map[code] == 1
    answer = next(iter(q.has_code(code).answer), None)
    if not answer:
        return False
    keys = answer.keys()
    return len(set(keys) - {"0", "9"}) > 3


# 변수가 연속변수인지 확인
def iscontinuous(q, code):
    from pykoges.__map import _type_map

    if code in _type_map:
        return _type_map[code] == 2
    return not isbinary(q, code) and not isdiscrete(q, code)


# 변수가 실수인지 확인
def isfloat(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


# 데이터를 저장하는 과정에 필요한 함수입니다.
def arr_to_df(df, column=None):
    import pandas as pd

    df = list(map(list, df))
    if column:
        df = pd.DataFrame(df, columns=column, index=None)
    else:
        df = pd.DataFrame(df[1:], columns=df[0], index=None)
    df = df.style.hide(axis="index")
    return df


def arr_to_df_split(df, n=5, column=None):
    import pandas as pd

    df = pd.DataFrame(df)
    if not column:
        column, df = df.iloc[0, :], df.iloc[1:, :]
    column = list(column) * ((len(df) + n - 1) // n)
    dfs = pd.DataFrame()
    for i in range(0, len(df), n):
        dfn = df.iloc[i : i + n, :]
        dfn = dfn.reset_index(drop=True)
        dfs = pd.concat([dfs, dfn], axis=1, ignore_index=True).fillna("-")
    dfs.columns = column
    dfs = dfs.style.hide(axis="index")
    return dfs


def df_to_img(df, path, title=None):
    import dataframe_image as dfi

    if title:
        df = df.set_caption(caption=title)
    if ".html" in path:
        with open(path, "w") as f:
            f.write(df.to_html().replace("\\n", "<br>"))
    else:
        dfi.export(df, path)


eps = 1e-9
# 변수, 함수에 맞춰 새로운 변수를 만들어줍니다. (차원축소/확장)
div = lambda a, b: a / (b + eps) * 100
mul = lambda a, b: a * b
