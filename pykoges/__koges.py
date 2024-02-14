__all__ = ["Variables", "KogesData", "read", "drop", "convert", "split_data"]


class Variables:
    def __init__(
        self,
        q,
        x_list={},
        y_list={},
        patientinfo_list={},
    ) -> None:
        x_list = list(set(x_list) - set(y_list))
        y_list = list(set(y_list))
        patientinfo_list = list(set(patientinfo_list))

        self.x = x_list
        self.y = y_list
        self.q = q
        self.patientinfo = patientinfo_list

    def summary(
        self,
        display_datainfo=True,
        display_userinfo=False,
    ) -> None:
        import numpy as np
        import pandas as pd
        from IPython.display import HTML, display

        key_list = self.q.keys()

        column_list = np.array([f"{d} {y}" for [d, y] in key_list])
        arr_data, arr_user = [], []

        # 질문 텍스트가 긴 경우 ...으로 표시
        def __long(s, l=4):
            return s[:l] + "..." if len(s) > l else s

        for [d, y] in key_list:
            # 입출력 변수에 포함되는 코드를 추출
            data = self.q.from_type(d, y).has_code(self.x + self.y)
            data = [
                x.survey_code + "\n" + __long(x.question_text) if x else None
                for x in data.list
            ]
            arr_data.append(data)

            # 환자 정보에 포함되는 코드를 추출
            user = self.q.from_type(d, y).has_code(self.patientinfo)
            user = [
                x.survey_code + "\n" + __long(x.question_text) if x else None
                for x in user.list
            ]
            arr_user.append(user)

        __index_map = {
            "신체계측": ["weight", "height", "bparmc", "waist", "hip"],
            "inbody": ["muscle", "incell", "excell", "pbf"],
            "inbody(골격근)": ["skmm", "axmm", "armrm", "armlm", "leglm", "legrm"],
            "호흡": ["fev1", "fvc", "fef25"],
            "순환": ["labi", "rabi", "pulse"],
            "뼈": ["stiffness", "bonet", "bonez"],
            "신장": ["bun", "creatine"],
            "CBC": ["rbc", "wbc", "plat", "hb", "hct", "mch", "mchc", "mcv"],
            "대사": ["alt", "ast", "hdl", "ldl", "r_gtp", "tchl", "tg", "t_bil"],
            "인지노화": ["grwhich", "gripl1", "gripr1"],
            "흡연음주": ["smam", "smdudy", "smdumo", "smduyr", "drinkam", "drinkfq"],
            "기본정보": ["age", "sex"],
        }
        __index = []
        for x in self.x + self.y:
            category = next((k for k, v in __index_map.items() if x in v), "other")
            __index.append([category, x])

        multi_index = pd.MultiIndex.from_tuples(__index)
        arr_data = np.array(arr_data).T
        arr_user = np.array(arr_user).T
        datainfo = (
            pd.DataFrame(
                arr_data,
                columns=column_list,
                index=multi_index,
            )
            .sort_index()
            .T
        )
        userinfo = pd.DataFrame(
            arr_user,
            columns=column_list,
            index=self.patientinfo,
        ).T
        datainfo = datainfo[datainfo.loc[:, (slice(None), self.y)].notna().all(axis=1)]
        datainfo = datainfo.dropna(axis=1, how="all").fillna("-")
        userinfo = userinfo.dropna(axis=1, how="all").fillna("-")
        if display_datainfo:
            print("입출력 변수 정보")
            datainfo = datainfo.style.set_table_styles(
                [
                    dict(
                        selector="th",
                        props=[
                            ("text-align", "center"),
                            ("border", "1px solid grey !important"),
                        ],
                    )
                ]
            )
            display(HTML(datainfo.to_html().replace("\\n", "<br>")))
        if display_userinfo:
            print("유저 변수 정보")
            display(HTML(userinfo.to_html().replace("\\n", "<br>")))


class KogesData:
    def __init__(
        self,
        variables: Variables = None,
    ):
        from pykoges.utils import isdiscrete, isbinary

        if not variables:
            return

        self.q = variables.q
        self.x = variables.x
        self.y = variables.y
        if isdiscrete(self.q, variables.y[0]):
            self.type = "discrete"
        elif isbinary(self.q, variables.y[0]):
            self.type = "binary"
        else:
            self.type = "continuous"
        self.patientinfo = variables.patientinfo
        self.SAVE = {}
        self.option = {}

    def copy(self):
        import copy

        res = self.__class__()
        for k, v in self.__dict__.items():
            try:
                setattr(res, k, copy.deepcopy(v))
            except:
                setattr(res, k, v)
        return res

    def save(
        self,
        isdisplay=True,
        issave=True,
    ):
        import os
        from pykoges.utils import name_map, arr_to_df_split, df_to_img
        from IPython.display import display
        from tqdm.notebook import tqdm
        from datetime import datetime as dt

        y = self.y[0]
        self.SAVE["time"] = dt.today().strftime("%y%m%d_%H%M")
        d = os.path.join("./result", f'{self.SAVE["time"]}_{y.upper()}')
        if not os.path.exists(d) and issave:
            os.makedirs(d, exist_ok=True)

        o = self.option
        keys = [
            "filter_patient",
            "drop_threshold",
            "data_impute",
            "filter_alpha",
            "muscle_weight_ratio",
            "muscle_height_ratio",
            "muscle_bmi_ratio",
            "waist_hip_ratio",
            "fev_fvc_ratio",
            "grip_of_grwhich",
            "abi_of_grwhich",
            "weight_height_bmi",
        ]
        option = [["이름", "옵션"], *[[k, o.get(k, "-")] for k in keys]]
        X = [name_map.get(x, x) for x in self.x]

        option = arr_to_df_split(option, n=3)
        inputs = arr_to_df_split(zip(self.x, X), column=["변수코드", "변수이름"])
        self.SAVE["option"] = option
        self.SAVE["input"] = inputs
        if isdisplay:
            print("-----------------")
            print("학습결과")
            display(option)
            print(f" - 입력변수 {len(X)}개")
            display(inputs)
        order = [
            "LinearRegression",
            "RandomForestRegressor",
            "LogisticRegression",
            #
            "DecisionTreeClassifier",
            "RandomForestClassifier",
            "softmaxClassifier",
            "multiclassRoc",
            #
            "option",
            "input",
            #
            "statistics",
            "boxplot",
            "correlation",
            "scatter",
            #
            "classes",
            "dropdata",
            "count",
            "importance",
        ]
        if "equation" in self.SAVE and issave:
            with open(os.path.join(d, "equation.tex"), "w", encoding="utf-8") as f:
                f.write(self.SAVE["equation"])
        i = 0
        indexes = {order.index(x): x for x in self.SAVE.keys() if x in order}
        keys = [x[1] for x in sorted(indexes.items())]
        if issave:
            for i in tqdm(range(len(keys)), desc="Saving result"):
                k = keys[i]
                v = self.SAVE[k]
                k = str(i + 1) + "." + k
                if not k.endswith(".png"):
                    k += ".png"
                c = v.__class__.__name__
                di = os.path.join(d, k)
                if c == "Styler" or c == "DataFrame":
                    df_to_img(v, di)
                elif c == "Figure":
                    v.savefig(di, bbox_inches="tight")


class kogesclass:
    @staticmethod
    def __drop_norm(df, q, right_alpha, left_alpha):
        import pandas as pd

        from pykoges.utils import iscontinuous

        df = pd.DataFrame(df)
        # 2SD를 벗어나는 데이터를 모두 제거합니다.
        for code in df:
            if iscontinuous(q, code):
                # 데이터 개수가 3개 이상이어야 데이터를 filtering할 수 있습니다.
                if len(df[code]) < 3:
                    continue
                df[code] = df[code].astype(float)
                m = df[code].mean()
                std = df[code].std()
                df = df[
                    (df[code] >= m - left_alpha * std)
                    & (df[code] <= m + right_alpha * std)
                ]
            # 연속 데이터가 아닌 경우
            else:
                df[code] = df[code].astype(int)
        return df

    def __drop_quantile(df, q, right_quantile, left_quantile):
        import pandas as pd
        from pykoges.utils import iscontinuous

        df = pd.DataFrame(df)
        for code in df:
            if iscontinuous(q, code):
                # 데이터 개수가 3개 이상이어야 데이터를 filtering할 수 있습니다.
                if len(df[code]) < 3:
                    continue
                df[code] = df[code].astype(float)
                df = df[
                    (df[code] >= df[code].quantile(left_quantile))
                    & (df[code] <= df[code].quantile(right_quantile))
                ]
            # 연속 데이터가 아닌 경우
            else:
                df[code] = df[code].astype(int)
        return df

    def read(
        variables: Variables,
        folder_name=None,
        filter_patient=True,
    ):
        import os
        import warnings

        import pandas as pd
        from tqdm.notebook import tqdm

        from pykoges.datatype import Patient, Patients
        from pykoges.utils import isbinary, isfloat

        _kg = KogesData(variables=variables)
        _kg.option["filter_patient"] = filter_patient

        if not len(_kg.y or []):
            raise ValueError("Y값으로 지정할 코드는 필수입력입니다.")
        # Question code, 질문 코드를 연도와 상관없이 모아줍니다.
        qcode = [_kg.q.has_code(code).survey_code for code in _kg.x]
        qcode = set(y for x in qcode for y in x)

        frames = {}
        patient_list = Patients([])

        # 시간 역순으로 데이터를 순회하여 중복되는 환자 데이터를 제거합니다.
        # (만약 2022년에 조사한 데이터가 있다면 2008년 데이터는 추가하지 않음)
        keys = _kg.q.keys()
        pbar = tqdm(keys)
        for data_type, year in pbar:
            key = " ".join([data_type, year])
            pbar.set_description(f"{key} 불러오는 중...")
            # baseline 08 데이터는 근육량이 아닌 골격근량을 측정하여 제외합니다.
            if "muscle" in _kg.x + _kg.y and data_type == "baseline" and year == "08":
                continue
            path = os.path.join(folder_name, f"data_{data_type}_{year}.csv")

            df = pd.read_csv(path, dtype=object)
            # 질문코드가 대문자로 되어있어 소문자로 변환해줍니다.
            df.columns = map(str.lower, df.columns)
            # 새로운 데이터 프레임을 생성합니다.
            ndf = pd.DataFrame()

            code_list = _kg.x + _kg.y
            if filter_patient:
                code_list += _kg.patientinfo
            for code in code_list:
                # check는 조건을 만족하는 질문 code가 데이터에 포함되었는지 여부입니다.
                check = False
                for x in df.columns:
                    # 조건은 질문코드가 원하는 값으로 끝나는지 여부 입니다.
                    if x.endswith(f"_{code}"):
                        ndf[code] = df[x]
                        if code not in _kg.patientinfo:
                            # 실수로 변환 가능한 데이터만 가져윰
                            ndf = ndf[ndf[code].apply(lambda x: isfloat(x))]
                            ndf[code] = ndf[code].astype(float)
                        check = True
                        break

                # 전체를 다 돌았음에도 질문 코드가 없었다면 None을 추가합니다.
                # 이 과정은 코드가 일부만 있는 데이터를 제거하기 위해 사용됩니다.
                if not check:
                    ndf[code] = None

            if filter_patient:
                del_rows = []
                for i, row in ndf.iterrows():
                    if row[_kg.patientinfo].any():
                        patient_dict = {k: row[k] for k in _kg.patientinfo}
                        patient = Patient(patient_dict)

                        if patient_list.has_patient(patient):
                            del_rows.append(i)
                        else:
                            patient_list.append(patient)
                ndf = ndf.loc[~ndf.index.isin(del_rows)]

            ndf = ndf[_kg.x + _kg.y]

            y_code = _kg.y[0]
            if y_code in ndf.columns:
                # 심전도 소견 결과
                if y_code in ["code1", "code2"]:
                    ndf = ndf[~ndf[y_code].isna()]
                    ndf[y_code] = ndf[y_code].astype(int)
                    # 0 = 검사 안함, 1 = WNL, 2 = nonspecific ST-T change
                    ndf = ndf[ndf[y_code] != 0]
                    ndf = ndf[ndf[y_code] != 1]
                    ndf = ndf[ndf[y_code] != 9999]
                elif y_code in ["locat1"]:
                    ndf[y_code] = ndf[y_code].astype(int)
                    ndf = ndf[ndf[y_code] != 1]
                # ekg와 dm등의 binary (0,1) 지표를 찾아줍니다.
                # 기준은 값의 종류가 5개 이내인 경우 (ex. 0,1,2,9) 로 하였습니다.
                # (0=X,1=O)인 데이터와 (1=X,2=O)인 데이터를 통일하기 위해 min()을 사용합니다.
                elif isbinary(_kg.q, y_code):
                    # nuchronic5 (골다공증) 과 같은 경우는 모든 데이터가 0으로 입력되어 있어 제거합니다.
                    if len(set(ndf[y_code])) == 1:
                        ndf[y_code] = None
                    else:
                        ndf = ndf[ndf[y_code] != 9]
                        if y_code == "ekg":
                            ndf = ndf[ndf[y_code] != 3]  # 3. missing
                        if y_code == "dm":
                            ndf = ndf[ndf[y_code] != 0]  # 0. 해당없음
                        ndf = ndf[~ndf[y_code].isna()]
                        ndf[y_code] = ndf[y_code].astype(int)
                        ndf[y_code] = (ndf[y_code] != ndf[y_code].min()).astype(int)

            # 데이터가 하나도 없는 경우는 제외
            ndf = ndf.dropna(axis=0, how="all")
            ndf = ndf.reset_index(drop=True)

            # 추가하려는 code를 모두 가진 데이터만 추가합니다.
            # 유효값을 거르는 과정입니다.
            if ndf.empty:
                continue
            frames[key] = ndf

        pbar.set_description("데이터 불러오기 완료")
        pbar.update(1)

        # 경고메시지가 안뜨도록 설정
        warnings.simplefilter(action="ignore", category=FutureWarning)
        df_read = pd.concat(frames)

        _kg.x = [x for x in df_read.columns if x != y_code]
        _kg.patient = patient_list
        _kg.data = df_read
        return _kg

    def convert(
        koges: KogesData,
        muscle_weight_ratio=False,
        muscle_height_ratio=False,
        muscle_bmi_ratio=False,
        waist_hip_ratio=False,
        fev_fvc_ratio=False,
        grip_of_grwhich=True,
        abi_of_grwhich=True,
        weight_height_bmi=False,
        appendicular_skeletal_muscle=True,
        custom_functions=[],
    ):
        import numpy as np
        import pandas as pd
        from .utils import div, mul, name_map, missing

        _kg = KogesData.copy(koges)
        df = pd.DataFrame(_kg.data)
        drop_list = []

        _kg.option["muscle_weight_ratio"] = muscle_weight_ratio
        _kg.option["muscle_height_ratio"] = muscle_height_ratio
        _kg.option["muscle_bmi_ratio"] = muscle_bmi_ratio
        _kg.option["waist_hip_ratio"] = waist_hip_ratio
        _kg.option["fev_fvc_ratio"] = fev_fvc_ratio
        _kg.option["grip_of_grwhich"] = grip_of_grwhich
        _kg.option["abi_of_grwhich"] = abi_of_grwhich
        _kg.option["weight_height_bmi"] = weight_height_bmi
        _kg.option["custom_functions"] = custom_functions
        # 코딩북 결측치 조건에 맞게 제거

        if "hdl" in df and "tg" in df and "tchl" in df:
            df["ldl"] = df["tchl"] - df["hdl"] - 0.2 * df["tg"]

        # 1. weight, height로 BMI를 계산합니다.
        if weight_height_bmi:
            if "weight" in df and "height" in df:
                df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
                drop_list += ["weight", "height"]
                if "weight" in _kg.y or "height" in _kg.y:
                    _kg.y = ["bmi"]

        # 2. 근육량을 체중 대비 비율로 변경합니다.
        if muscle_weight_ratio:
            if "weight" in df and "muscle" in df:
                df["muscle_weight"] = df["muscle"] / df["weight"]
                drop_list += ["weight", "muscle"]
                if "weight" in _kg.y or "muscle" in _kg.y:
                    _kg.y = ["muscle_weight"]
        elif muscle_height_ratio:
            if "height" in df and "muscle" in df:
                df["muscle_height"] = df["muscle"] / df["height"]
                drop_list += ["height", "muscle"]
                if "height" in _kg.y or "muscle" in _kg.y:
                    _kg.y = ["muscle_height"]
        elif muscle_bmi_ratio:
            if "bmi" in df and "muscle" in df:
                df["muscle_bmi"] = df["muscle"] / df["bmi"]
                drop_list += ["bmi", "muscle"]
                if "bmi" in _kg.y or "muscle" in _kg.y:
                    _kg.y = ["muscle_bmi"]

        # 3. whr을 계산해 추가합니다.
        if waist_hip_ratio:
            if "waist" in df and "hip" in df:
                df["whr"] = df["waist"] / df["hip"]
                drop_list += ["waist", "hip"]
                if "waist" in _kg.y or "hip" in _kg.y:
                    _kg.y = ["whr"]

        # 4. fev1/fvc를 계산해 추가합니다.
        if fev_fvc_ratio:
            if "fev1" in df and "fvc" in df:
                df["fev1fvc"] = df["fev1"] / df["fvc"]
                drop_list += ["fev1", "fvc"]
                if "fev1" in _kg.y or "fvc" in _kg.y:
                    _kg.y = ["fev1fvc"]

        # 5. 자주 사용하는 손 방향의 악력과 ABI만을 가져옵니다.
        if grip_of_grwhich:
            if "gripl1" in df and "gripr1" in df and "grwhich" in df:
                isright = df["grwhich"] == df["grwhich"].min()
                df["grip"] = np.where(isright, df["gripr1"], df["gripl1"])
                drop_list += ["gripr1", "gripl1", "grwhich"]
                if "gripl1" in _kg.y or "grwhich" in _kg.y:
                    _kg.y = ["grip"]
                    _kg.type = "continuous"
        if abi_of_grwhich:
            if "grwhich" in df and "labi" in df and "rabi" in df:
                isright = df["grwhich"] == df["grwhich"].min()
                df["abi"] = np.where(isright, df["rabi"], df["labi"])
                drop_list += ["labi", "rabi", "grwhich"]
                if "labi" in _kg.y and "rabi" in _kg.y:
                    _kg.y = ["abi"]
                    _kg.type = "continuous"

        if appendicular_skeletal_muscle:
            if "armrm" in df and "armlm" in df and "legrm" in df and "leglm" in df:
                df["asm"] = df["armrm"] + df["armlm"] + df["legrm"] + df["leglm"]
                drop_list += ["armrm", "armlm", "legrm", "leglm"]
                if (
                    "armrm" in _kg.y
                    and "armlm" in _kg.y
                    and "legrm" in _kg.y
                    and "leglm" in _kg.y
                ):
                    _kg.y = ["asm"]
                    _kg.type = "continuous"

        # 흡연, 음주
        if "smam" in df:
            df = df[(df["smam"] != 99)]
            if "smdudy" in df and "smdumo" in df and "smduyr" in df:
                df = df[
                    (df["smdudy"] != 99) & (df["smdumo"] != 99) & (df["smduyr"] != 99)
                ]
                df["smokepy"] = (df["smam"] / 20) * (
                    df["smdudy"] / 365 + df["smdumo"] / 12 + df["smduyr"]
                )
                drop_list += ["smam", "smdudy", "smdumo", "smduyr"]
            elif "smdu" in df:
                df = df[(df["smdu"]) != 99]
                df["smokepy"] = (df["smam"] / 20) * (df["smdu"])
                drop_list += ["smam", "smdu"]
        if "drinkfq" in df and "drinkam" in df:
            df = df[(df["drinkfq"] != 9) & (df["drinkam"] != 999)]
            df["drinkfq"] = df["drinkfq"].replace(
                {1: 0.25, 2: 0.42, 3: 1, 4: 2.5, 5: 5, 6: 7, 7: 14}
            )
            df["drinkaw"] = df["drinkfq"] * df["drinkam"]
            drop_list += ["drinkfq", "drinkam"]

        # 6. 기타 함수로 나타낼 항목
        for x, f in custom_functions:
            if set(x).issubset(df.columns):
                x_name = [name_map.get(y, y) for y in x]
                if f == mul:
                    c = f'({"*".join(x_name)})'
                elif f == div:
                    c = f'({"/".join(x_name)})'
                else:
                    c = f'f({",".join(x_name)})'
                df[c] = f(*[df[x] for x in x])
                drop_list += x
                if set(x).issubset(_kg.y):
                    _kg.y = [c]

        _kg.drop_list = drop_list
        _kg.data = df
        return _kg

    def drop(
        koges: KogesData,
        drop_threshold=0.3,
        filter_left_alpha=float("inf"),
        filter_right_alpha=float("inf"),
        filter_right_quantile=1,
        filter_left_quantile=0,
        data_impute=False,
        display_result=True,
        display_count=True,
    ):
        import pandas as pd
        from IPython.display import HTML, display
        from sklearn.impute import KNNImputer

        from .datatype import Question
        from .utils import arr_to_df, missing

        _kg = KogesData.copy(koges)
        df = pd.DataFrame(_kg.data)

        _kg.option["drop_threshold"] = drop_threshold
        _kg.option["filter_alpha"] = filter_right_alpha
        _kg.option["data_impute"] = data_impute
        # isbinary나 isdiscrete에서 오류가 나지 않도록
        # 새로 생성된 변수에 대해 Questions에 추가해줍니다.
        for x in df:
            if not _kg.q.has_code(x).len:
                question = Question(survey_code=f"_{x}")
                _kg.q.list.append(question)

        # 결측치를 KNN알고리즘으로 채워줍니다.
        if data_impute:
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index,
            )

        y = _kg.y[0]
        df_y = df[~df[y].isna()]
        # drop_threshold 이상의 비율의 결측치를 가진 변수를 제외
        df_var = df_y.loc[:, df_y.isnull().mean() < drop_threshold]
        # 변수를 하나라도 가지지 않은 경우 제거
        df_drop = df_var.dropna(axis=0, how="any")
        df_drop = df_drop.dropna(axis=1, how="all")
        # 코딩북 조건에 맞게 결측치 제거
        n = len(df_drop)
        for x in df_drop.columns:
            v = missing.get(x, False)
            if v:
                if isinstance(v, list):
                    df_drop = df_drop[~df_drop[x].astype(float).isin(v)]
                else:
                    df_drop = df_drop[df_drop[x].astype(float) != v]
        print(f"코딩북 결측치 조건에 맞게 {n-len(df_drop)}개 데이터 제거됨.")
        df_drop = df_drop.drop(set(_kg.drop_list), axis=1)
        # dropNorm으로 정규분포를 벗어나는 데이터 제거
        df_sdfilter = kogesclass.__drop_norm(
            df_drop, _kg.q, right_alpha=filter_right_alpha, left_alpha=filter_left_alpha
        )
        # dropQuantile으로 상하위 % 데이터 제거
        df_filter = kogesclass.__drop_quantile(
            df_sdfilter,
            _kg.q,
            right_quantile=filter_right_quantile,
            left_quantile=filter_left_quantile,
        )
        n, n1, n2, n3 = len(df), len(df_y), len(df_drop), len(df_filter)

        # 결측치를 처리한 경우 제거된 결측치가 없으므로 출력하지 않습니다.
        result = [
            ["", "N", "남성", "여성", "비율", "변수"],
            [
                "전체 데이터",
                n,
                len(df[df["sex"] == "1"]),
                len(df[df["sex"] == "2"]),
                "100%",
                len(df.columns) - len(set(_kg.drop_list)),
            ],
            [
                "Y값 결측치 제거",
                n1 - n,
                "",
                "",
                f"{int((n1-n)/n*100)}%",
                (len(df_y.columns) - len(df.columns)) or "",
            ],
            [
                f"결측치 {int(drop_threshold*100)}% 이상인\n입력변수 제거",
                "",
                "",
                "",
                "",
                (len(df_var.columns) - len(df_y.columns)) or "",
            ],
            [
                "결측치 제거",
                n2 - n1,
                "",
                "",
                f"{int((n2-n1)/n*100)}%",
                (len(df_drop.columns) - len(df_var.columns) + len(set(_kg.drop_list)))
                or "",
            ],
            (filter_right_alpha != float("inf") or filter_left_alpha != float("inf"))
            and [
                f"+{filter_right_alpha}SD 초과, -{filter_left_alpha}SD 미만제거",
                n3 - n2,
                "",
                "",
                f"{int((n3-n2)/n*100)}%",
                (len(df_sdfilter.columns) - len(df_drop.columns)) or "",
            ],
            [
                "최종데이터",
                n3,
                len(df_filter[df_filter["sex"] == 1]),
                len(df_filter[df_filter["sex"] == 2]),
                f"{int(n3/n*100)}%",
                len(df_filter.columns),
            ],
        ]
        result = arr_to_df([x for x in result if x])
        result = result.set_table_styles(
            [dict(selector="th", props=[("white-space", "pre")])]
        )

        if display_result:
            display(HTML(result.to_html()))

        # 데이터가 없으면 error를 띄워 프로그램 진행을 멈춥니다.
        if df_filter.empty:
            raise ValueError(
                "조건을 만족하는 데이터가 존재하지 않습니다.\ndrop_threshold를 더 낮게 조정하세요."
            )

        keys = _kg.q.keys(astype=str)
        count = [df_filter.index.isin([key], level=0).sum() for key in keys]
        count = pd.DataFrame(count, index=keys, columns=["데이터 개수"])
        count = count[count.iloc[:, 0] != 0].T
        count["total "] = count.sum(axis=1)
        count = count.T
        count.index = pd.MultiIndex.from_tuples(
            [tuple(str.split(x, " ")) for x in count.index]
        )
        if display_count:
            display(count)

        _kg.x = [x for x in df_filter.columns if x != y]
        _kg.SAVE["dropdata"] = result
        _kg.SAVE["count"] = count
        _kg.data = df_filter
        return _kg

    @staticmethod
    def __filter_class(
        koges,
        n_class,
        convert={},
        isdisplay=True,
    ):
        import json
        import pandas as pd
        from IPython.display import display

        from pykoges.utils import isdiscrete

        _kg = koges
        y = _kg.y[0]
        # 상위 5개 class만 추출
        if not isdiscrete(_kg.q, y):
            return
        df = pd.DataFrame(_kg.data)
        codes = set(df[y].astype(str))
        # 답변을 코드:답변내용과 답변내용:코드로 모아줍니다.
        ans, ans_rev = {}, {}
        for answer in _kg.q.has_code("code1").answer:
            for k, v in json.loads(answer).items():
                if not isinstance(k, int) and not k.isnumeric():
                    continue
                ans[int(k)] = v
                ans_rev[v] = int(k)
        df_all = pd.DataFrame(columns=["응답내용", "n"])

        # 만약 답변내용이 변환맵에 존재하면
        for code, _from in ans.items():
            if _from in convert:
                # 변환값이 답변내용에 없다면 코드를 새로 생성
                _to = convert[_from]
                if _to not in ans_rev:
                    i = max(ans_rev.values()) + 1
                    ans_rev[_to] = i
                    # q.has_code('code1').list[1].answer[i] = integrate[v]
                # 새로 생성했거나 원래 존재하는 변환값의 코드
                ans_rev[_from] = ans_rev[_to]
                nk = ans_rev[_to]
                # 기존의 코드를 모두 새 코드로 변환
                df[y] = df[y].astype(int).replace(int(code), int(nk))
        codes = set(df[y])

        for code in codes:
            for v, k in ans_rev.items():
                if k == code:
                    count = (df[y] == code).sum()
                    df_all.loc[k] = [convert.get(v, v), count]

        n_class = min(len(df_all), n_class)
        # 상위 n개만 추출 (기본값 5)
        df_all = df_all.nlargest(n_class, "n")
        df_all["%"] = (df_all["n"] / df_all["n"].sum() * 100).round(2).astype(str)
        class_map = {x: i for i, x in enumerate(df_all.index)}

        # 응답내용 리스트
        classes = list(df_all.iloc[:, 0])
        # 상위 n개만 추출
        df = df[df[y].isin(df_all.index)]
        # 응답 코드를 0,1,2,....로 변경
        df.loc[:, y] = df[y].replace(class_map)
        df_all = df_all.style.hide(axis="index")
        if isdisplay:
            print("-----------------")
            print(f"상위 {n_class}개 소견 {len(df)}개")
            display(df_all)

        _kg.SAVE["classes"] = df_all
        _kg.data = df
        _kg.n_class = n_class
        _kg.classes = classes
        return _kg

    def split_data(
        koges,
        n_class=4,
        isdisplay=True,
        custom_split={},
    ):
        import pandas as pd
        from pykoges.utils import ekg_map, name_map

        _kg = KogesData.copy(koges)
        y = _kg.y[0]
        datas = {}
        col_r1 = []
        if _kg.type == "binary" or _kg.type == "continuous":
            df = pd.DataFrame(_kg.data)
            # binary 혹은 연속변수
            if _kg.type == "binary":
                n_class = 2
                # binary는 양성/음성으로 분리
                classes = ["(-)", "(+)"]
                datas[0], datas[1] = df[df[y] == 0], df[df[y] == 1]
            elif custom_split:
                n_class = len(custom_split.keys())
                classes = list(custom_split.keys())

                df_remain = pd.DataFrame(df)
                for i, f in enumerate(custom_split.values()):
                    datas[i], df_remain = (
                        df_remain[f(df_remain)],
                        df_remain[~f(df_remain)],
                    )
            else:
                n_class = n_class or 4
                # 연속은 quantile을 기준으로 분리
                classes = [f"Q{i+1}" for i in range(n_class)]
                for i in range(n_class):
                    datas[i] = pd.DataFrame(
                        df[
                            (df[y] >= df[y].quantile(1 / n_class * i))
                            & (df[y] < df[y].quantile(1 / n_class * (i + 1)))
                        ]
                    )
                    datas[i][y] = i
            for i in range(n_class):
                # quan = df[y].quantile(1 / n_class * (i + 1))
                n = len(datas[i])
                col_r1 += [f"{classes[i]}\n{n}({n/len(df)*100:.2f})"]
            _kg.classes = classes
            _kg.n_class = n_class
        else:
            # multiclass
            convert = {}
            if y == "code1":
                convert = ekg_map
            # 상위 n_class개 만큼의 데이터만 추출합니다.
            _kg = kogesclass.__filter_class(
                koges=_kg,
                n_class=n_class,
                convert=convert,
                isdisplay=isdisplay,
            )
            df = pd.DataFrame(_kg.data)
            for i, x in enumerate(_kg.classes):
                datas[i] = df[df[y] == i]
                n = len(datas[i])
                col_r1 += [f"{name_map.get(x,x)}\n{n}({n/len(df)*100:.2f})"]
        # col_r2 = ["평균", "95%CI"] * n_class + ["p-value"]
        col_r2 = ["평균"] * n_class + [""]

        _kg.columns = [col_r1 + ["p"], col_r2]
        _kg.datas = datas
        return _kg
