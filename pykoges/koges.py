# 변수가 실수인지 확인
def isfloat(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


class koges:
    from pykoges.datatype import Questions

    def __init__(
        self,
        q=Questions(),
        x_list={},
        y_list={},
        patientinfo_list={},
    ) -> None:
        from datetime import datetime as dt

        x_list = list(x_list - y_list)
        y_list = list(y_list)
        patientinfo_list = list(patientinfo_list)

        self.x = x_list
        self.y = y_list
        self.q = q
        self.patientinfo = patientinfo_list
        self.SAVE = {
            "time": dt.today().strftime("%y%m%d_%H%M"),
        }

    def summary(
        self,
        print_datainfo=True,
        print_userinfo=False,
    ):
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
            "호흡": ["fev1", "fvc", "fef25"],
            "순환": ["labi", "rabi", "pulse"],
            "뼈": ["stiffness", "bonet", "bonez"],
            "신장": ["bun", "creatine"],
            "CBC": ["rbc", "wbc", "plat", "hb", "hct", "mch", "mchc", "mcv"],
            "대사": ["alt", "ast", "hdl", "ldl", "r_gtp", "tchl", "tg", "t_bil"],
            "인지노화": ["grwhich", "gripl1", "gripr1"],
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
        if print_datainfo:
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
        if print_userinfo:
            print("유저 변수 정보")
            display(HTML(userinfo.to_html().replace("\\n", "<br>")))

    # 변수가 이진변수인지 확인
    def isbinary(self, code):
        answer = next(self.q.has_code(code).answer, None)
        if not answer:
            return False
        # O, X, 무응답
        keys = answer.keys()
        return 0 < len(set(keys) - {"0", "9"}) <= 3

    # 변수가 이산변수인지 확인
    def isdiscrete(self, code):
        answer = next(self.q.has_code(code).answer, None)
        if not answer:
            return False
        keys = answer.keys()
        return len(set(keys) - {"0", "9"}) > 3

    # 변수가 연속변수인지 확인
    def iscontinuous(self, code):
        return not self.isbinary(code) and not self.isdiscrete(code)

    def read(
        self,
        folderName,
        filter_patient=True,
    ):
        from pykoges.datatype import Patients, Patient
        from tqdm.notebook import tqdm
        import pandas as pd
        import os

        if not len(self.y or []):
            raise ValueError("Y값으로 지정할 코드는 필수입력입니다.")
        # Question code, 질문 코드를 연도와 상관없이 모아줍니다.
        qcode = [self.q.has_code(code).survey_code for code in self.x]
        qcode = set(y for x in qcode for y in x)

        frames = {}
        patient_list = Patients([])

        # 시간 역순으로 데이터를 순회하여 중복되는 환자 데이터를 제거합니다.
        # (만약 2022년에 조사한 데이터가 있다면 2008년 데이터는 추가하지 않음)
        keys = self.q.keys()
        pbar = tqdm(keys)
        for data_type, year in pbar:
            key = " ".join([data_type, year])
            pbar.set_description(f"{key} 불러오는 중...")
            # baseline 08 데이터는 근육량이 아닌 골격근량을 측정하여 제외합니다.
            if "muscle" in self.x + self.y and data_type == "baseline" and year == "08":
                continue
            path = os.path.join(folderName, f"data_{data_type}_{year}.csv")

            df = pd.read_csv(path, dtype=object)
            # 질문코드가 대문자로 되어있어 소문자로 변환해줍니다.
            df.columns = map(str.lower, df.columns)
            # 새로운 데이터 프레임을 생성합니다.
            ndf = pd.DataFrame()

            code_list = self.x + self.y
            if filter_patient:
                code_list += self.patientinfo
            for code in code_list:
                # check는 조건을 만족하는 질문 code가 데이터에 포함되었는지 여부입니다.
                check = False
                for x in df.columns:
                    # 조건은 질문코드가 원하는 값으로 끝나는지 여부 입니다.
                    if x.endswith(f"_{code}"):
                        ndf[code] = df[x]
                        if code not in self.patientinfo:
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
                    if row[self.patientinfo].any():
                        patient_dict = {k: row[k] for k in self.patientinfo}
                        patient = Patient(patient_dict)

                        if patient_list.has_patient(patient):
                            del_rows.append(i)
                        else:
                            patient_list.append(patient)
                ndf = ndf.loc[~ndf.index.isin(del_rows)]

            ndf = ndf[self.x + self.y]

            y_code = self.y[0]
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
                elif self.isbinary(y_code):
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
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)
        df_read = pd.concat(frames)
        self.x = [x for x in df_read.columns if x != y_code]
        self.patient = patient_list
        self.data = df_read

    def __dropNorm(self, alpha=2):
        import pandas as pd

        df = pd.DataFrame(self.data)
        # 2SD를 벗어나는 데이터를 모두 제거합니다.
        for code in df:
            if self.iscontinuous(code):
                # 데이터 개수가 3개 이상이어야 데이터를 filtering할 수 있습니다.
                if len(df[code]) < 3:
                    continue
                df[code] = df[code].astype(float)
                m = df[code].mean()
                std = df[code].std()
                df = df[(df[code] >= m - alpha * std) & (df[code] <= m + alpha * std)]
            # 연속 데이터가 아닌 경우
            elif not self.iscontinuous(code):
                df[code] = df[code].astype(int)
        return df

    def drop(
        self,
        drop_threshold=0.3,
        filter_alpha=2,
        data_impute=False,
        muscle_weight_ratio=False,
        muscle_height_ratio=False,
        muscle_bmi_ratio=False,
        waist_hip_ratio=False,
        fev_fvc_ratio=False,
        grip_of_grwhich=True,
        weight_height_bmi=False,
        custom_function=[],
    ):
        from pykoges.datatype import Question
        import pandas as pd

        df = pd.DataFrame(self.data)
        drop_list = []

        # 1. weight, height로 BMI를 계산합니다.
        if weight_height_bmi:
            if "weight" in df and "height" in df:
                df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
                drop_list += ["weight", "height"]
                if "weight" in self.y or "height" in self.y:
                    self.y = ["bmi"]

        # 2. 근육량을 체중 대비 비율로 변경합니다.
        if muscle_weight_ratio:
            if "weight" in df and "muscle" in df:
                df["muscle_weight"] = df["muscle"] / df["weight"]
                drop_list += ["weight", "muscle"]
                if "weight" in self.y or "muscle" in self.y:
                    self.y = ["muscle_weight"]
        elif muscle_height_ratio:
            if "height" in df and "muscle" in df:
                df["muscle_height"] = df["muscle"] / df["height"]
                drop_list += ["height", "muscle"]
                if "height" in self.y or "muscle" in self.y:
                    self.y = ["muscle_height"]
        elif muscle_bmi_ratio:
            if "bmi" in df and "muscle" in df:
                df["muscle_bmi"] = df["muscle"] / df["bmi"]
                drop_list += ["bmi", "muscle"]
                if "bmi" in self.y or "muscle" in self.y:
                    self.y = ["muscle_bmi"]

        # 3. whr을 계산해 추가합니다.
        if waist_hip_ratio:
            if "waist" in df and "hip" in df:
                df["whr"] = df["waist"] / df["hip"]
                drop_list += ["waist", "hip"]
                if "waist" in self.y or "hip" in self.y:
                    self.y = ["whr"]

        # 4. fev1/fvc를 계산해 추가합니다.
        if fev_fvc_ratio:
            if "fev1" in df and "fvc" in df:
                df["fev1fvc"] = df["fev1"] / df["fvc"]
                drop_list += ["fev1", "fvc"]
                if "fev1" in self.y or "fvc" in self.y:
                    self.y = ["fev1fvc"]

        # 5. 자주 사용하는 손의 악력만을 가져옵니다.
        if grip_of_grwhich:
            if "gripl1" in df and "gripr1" in df and "grwhich" in df:
                isright = df["grwhich"] == df["grwhich"].min()
                df["grip"] = np.where(isright, df["gripr1"], df["gripl1"])
                drop_list += ["gripr1", "gripl1", "grwhich"]
                if "gripl1" in self.y or "grwhich" in self.y:
                    self.y = ["grip"]

        # 6. 기타 함수로 나타낼 항목
        for x, f in custom_function:
            if set(x).issubset(df.columns):
                x_name = [_name_map.get(y, y) for y in x]
                if f == mul:
                    c = f'({"*".join(x_name)})'
                elif f == div:
                    c = f'({"/".join(x_name)})'
                else:
                    c = f'f({",".join(x_name)})'
                df[c] = f(*[df[x] for x in x])
                drop_list += x
                if set(x).issubset(self.y):
                    self.y = [c]
        df = df.drop(set(drop_list), axis=1)

        # isbinary나 isdiscrete에서 오류가 나지 않도록
        # 새로 생성된 변수에 대해 Questions에 추가해줍니다.
        for x in df:
            if not self.q.has_code(x).len:
                question = Question(survey_code=f"_{x}", answer=[])
                self.q.list.append(question)

        # 결측치를 KNN알고리즘으로 채워줍니다.
        if data_impute:
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(
                imputer.fit_transform(df), columns=df.columns, index=df.index
            )

        y_code = self.y[0]
        # y값이 결측치인 데이터를 제외
        df_nna = df[df[y_code].notna()]
        # drop_threshold 이상의 비율의 결측치를 가진 변수를 제외
        df_drop_var = df_nna.loc[:, df_nna.isnull().mean() >= drop_threshold]
        df_var = df_nna.loc[:, df_nna.isnull().mean() < drop_threshold]
        # 변수를 하나라도 가지지 않은 경우 제거
        df_drop = df_var.dropna(axis=0, how="any")
        # dropNorm으로 정규분포를 벗어나는 데이터 제거
        df_sdfilter = self.__dropNorm(df_drop, alpha=filter_alpha)
        n, n1, n2, n3 = len(df), len(df_nna), len(df_drop), len(df_sdfilter)

        # 결측치를 처리한 경우 제거된 결측치가 없으므로 출력하지 않습니다.
        result = [
            ["전체 데이터", n, "100%", len(df.columns)],
            [
                "출력값 결측치 제거",
                n1 - n,
                f"{int((n1-n)/n*100)}%",
                len(df_nna.columns),
            ],
            [
                "입력변수 제거",
                "",
                "",
                # f'{int(drop_threshold*100)}% 이상의 데이터가\n결측치인 변수 {len(df_drop_var.columns)}개',
                len(df_nna.columns) - len(df_drop_var.columns),
            ],
            [
                "결측치 제거",
                n2 - n1,
                f"{int((n2-n1)/n*100)}%",
                len(df_drop.columns),
            ],
            [
                f"{filter_alpha}SD 초과제거",
                n3 - n2,
                f"{int((n3-n2)/n*100)}%",
                len(df_sdfilter.columns),
            ],
            ["최종데이터", n3, f"{int(n3/n*100)}%", len(df_sdfilter.columns)],
        ]
        result = arr_to_df(result, column=["수행", "데이터", "비율", "변수"])
        display(result)

        self.x = [x for x in df_sdfilter.columns if x != y_code]
        # 데이터가 없으면 error를 띄워 프로그램 진행을 멈춥니다.
        if df_sdfilter.empty:
            raise ValueError("조건을 만족하는 데이터가 존재하지 않습니다.\ndrop_threshold를 더 낮게 조정하세요.")

        keys = self.q.keys(astype=str)
        count = [df_sdfilter.index.isin([key], level=0).sum() for key in keys]
        count = pd.DataFrame(count, index=keys, columns=["데이터 개수"])
        count = count[count.iloc[:, 0] != 0].T
        count["total "] = count.sum(axis=1)
        count = count.T
        count.index = pd.MultiIndex.from_tuples(
            [tuple(str.split(x, " ")) for x in count.index]
        )
        display(count)

        self.SAVE["dropdata"] = result
        self.SAVE["count"] = count
        return df_sdfilter
