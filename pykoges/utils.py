__all__ = [
    "isbinary",
    "isdiscrete",
    "iscontinuous",
    "isfloat",
    "arr_to_df",
    "arr_to_df_split",
    "df_to_img",
    "div",
    "mul",
    "name_map",
    "ekg_map",
    "type_map",
]


# 변수가 이진변수인지 확인
def isbinary(q, code):
    import json

    if code in type_map:
        return type_map[code] == 0
    answer = next(iter(q.has_code(code).answer), None)
    if not answer:
        return False
    # O, X, 무응답
    keys = json.loads(answer).keys()
    return 0 < len(set(keys) - set(["0", "9"])) <= 3


# 변수가 이산변수인지 확인
def isdiscrete(q, code):
    import json

    if code in type_map:
        return type_map[code] == 1
    answer = next(iter(q.has_code(code).answer), None)
    if not answer:
        return False
    keys = json.loads(answer).keys()
    return len(set(keys) - set(["0", "9"])) > 3


# 변수가 연속변수인지 확인
def iscontinuous(q, code):
    if code in type_map:
        return type_map[code] == 2
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
    if not column:
        column, df = df[0], df[1:]
    df = pd.DataFrame(df, columns=column, index=None)
    df = df.style.hide(axis="index")
    return df


def arr_to_df_split(df, n=5, column=None):
    import pandas as pd

    df = list(map(list, df))
    if not column:
        column, df = df[0], df[1:]
    column = list(column) * ((len(df) + n - 1) // n)
    dfs = pd.DataFrame()
    for i in range(0, len(df), n):
        dfn = pd.DataFrame(df[i : i + n])
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


def get_first_col(df):
    if len(df.shape) > 1:
        if df.shape[1] > 1:
            df = df.iloc[:, 1]
    return df


def remove_duplicate_col(df):
    import pandas as pd

    df = pd.DataFrame(df)
    return df.loc[:, ~df.columns.duplicated()].copy()


eps = 1e-9
# 변수, 함수에 맞춰 새로운 변수를 만들어줍니다. (차원축소/확장)
div = lambda a, b: a / (b + eps) * 100
mul = lambda a, b: a * b


name_map = {
    "age": "Age",
    "albumin": "Albumin",
    "alt": "ALT",
    "ast": "AST",
    "bonet": "T-score",
    "bonez": "Z-score",
    "bparmc": "팔둘레",
    "bmi": "BMI",
    "bun": "BUN",
    "creatine": "Creatine",
    "crp": "CRP",
    "fef25": "FEF25",
    "fev1": "FEV1",
    "fvc": "FVC",
    # "gripl1": "",
    "gripr1": "악력",
    "grip": "악력",
    # "grwhich": "",
    "hb": "Hb",
    "hct": "Hct",
    "hdl": "HDL",
    "hba1c": "HbA1C",
    "glu0": "혈당",
    "height": "키",
    "hip": "엉덩이 둘레",
    "waist": "허리둘레",
    "abi": "ABI",
    "labi": "ABI(왼쪽)",
    "rabi": "ABI(오른쪽)",
    "ldl": "LDL",
    "mch": "MCH",
    "mchc": "MCHC",
    "mcv": "MCV",
    "muscle": "SLM",
    "skeletal_muscle": "골격근량",
    "plat": "PLT",
    "pulse": "맥박",
    "r_gtp": "γ-GTP",
    "rbc": "RBC",
    "stiffness": "Stiffness",
    "t_bil": "T.Bilirubin",
    "tchl": "Cholesterol",
    "tg": "TG",
    "u_ket": "U.ketone",
    "u_ph": "U.pH",
    "uricacid": "UA",
    "wbc": "WBC",
    "whr": "WHR",
    "weight": "체중",
    "ekg": "EKG",
    "code1": "EKG소견",
    "dm": "당뇨",
    "smokepy": "흡연량(갑년)",
    "drinkaw": "주당음주량",
    # 임의추가
    "skmm": "골격근량",
    "axmm": "Axial muscle",
    "armrm": "오른팔근육량",
    "armlm": "왼팔근육량",
    "legrm": "오른다리근육량",
    "leglm": "왼다리근육량",
    "asm": "ASM",
}

ekg_map = {
    "RAE": "RAH",
    "LAE": "LAH",
    "LVH by voltage criteria": "LVH",
    "LVH by cornell voltage": "LVH",
    "LVH with strain": "LVH",
    # A.fib.
    "A.fib.": "A fib",
    "A.fib.": "A fib",
    "A fib with controlled VR": "A fib",
    "A fib with rapid VR": "A fib",
    "A fib with slow VR": "A fib",
    "A.fib. With controlled VR": "A fib",
    "A.fib. With rapid VR": "A fib",
    "A.fib. With slow VR": "A fib",
    "A.fib.with marked slow VR": "A fib",
    # premature
    "APC": "premature",
    "APC bigeminy": "premature",
    "APC or JPC": "premature",
    "VPC": "premature",
    "VPC bigeminy": "premature",
    "VPC couplet": "premature",
    "VPC trigeminy": "premature",
    "VPCs quadrigeminy": "premature",
    "VPCs quardisgeminy": "premature",
    # pre-excitation
    "LGL syndrome": "pre-excitation",
    "preexcitation": "pre-excitation",
    "preexitation": "pre-excitation",
    "WPW syndrome": "pre-excitation",
    "WPW sydrome": "pre-excitation",
    "ventricular preexcitation": "pre-excitation",
    "ventricular preexitation": "pre-excitation",
    "ventriculat preexitation": "pre-excitation",
    # MI
    "old MI anteroseptal": "MI",
    "old MI inferior": "MI",
    "old MI lateral": "MI",
    "old MI posterior": "MI",
    "old MI septal": "MI",
    "old MI, anterior": "MI",
    "old MIf, anterior(QS pattern in all of leads V1, V2, and V3)": "MI",
    "old MIf, anterior(QS pattern in all of leads V1-V4 or V1-V5)": "MI",
    "old Mif anteroseptal": "MI",
    "old Mif inferior": "MI",
    "old Mif lateral": "MI",
    "old Mif posterior": "MI",
    "old Mif septal": "MI",
    "myocardial injury anterior wall": "MI",
    "myocardial ischemia": "MI",
    "myocardial ischemia(MIs)(anterior  wall)": "MI",
    "myocardial ischemia(MIs)(anteroseptal wall)": "MI",
    "myocardial ischemia(MIs)(inferior wall)": "MI",
    "MIs(mycardial ischemia)": "MI",
    "MIs(myocardial ischemia)(lateral ischemia)": "MI",
    "Mif(mycardial infarction)": "MI",
    "r/o) old MI(posterior wall)": "MI",
    "r/o) old MIf(myocardial infarction)(posterior wall)": "MI",
    "extensive anterior MI": "MI",
    "extensive anterior MIf": "MI",
    # sinus arrhythmia
    "sinus bradycardia": "sinus arrhythmia",
    "sinus bradycardia with junctional escape beat": "sinus arrhythmia",
    "sinus tachycardia": "sinus arrhythmia",
    "marked sinus bradycardia": "sinus arrhythmia",
    "marked sinus bradycardia with junctional escape beat": "sinus arrhythmia",
    # BBB
    "LBBB": "BBB",
    "RBBB": "BBB",
    "incomplete LBBB": "BBB",
    "incomplete RBBB": "BBB",
    "complete LBBB": "BBB",
    "complete RBBB": "BBB",
    # Heart block
    "AV dissociation": "block",
    "High grade AV block": "block",
    "complete AV block": "block",
    "first degree AV block": "block",
    "secondary degree AV block (type 1 -partial)": "block",
    "Advanced AV block(2:1 conduction)": "block",
    "Complete  heart block with junctinal rhythem": "block",
    "Complete  heart block with junctional rhythem": "block",
    "bifascicular block": "block",
    "left anterior hemiblock": "block",
    "sinoatrial block": "block",
    "sionatrial block": "block",
    "trigascifulcar black(三支阻滞）": "block",
    "LPFB": "block",
    # arrest
    "sinus arrest with escape beats": "arrest",
    "sinus arrest with junctional escape": "arrest",
    "sinus arrest with nunctional escape": "arrest",
    "sinus arrest with wandering atrial pacemaker": "arrest",
    # AFL
    "atrial Filtter with 4:1 conduction": "AFL",
    "atrial filtter with variable conduction": "AFL",
    "atrial flutter with 4:1 conduction": "AFL",
    "atrial flutter with variable conduction": "AFL",
    # ST-T change
    "brugada pattern": "ST-T change",
    "nonspecific ST-T change": "ST-T change",
    "r/o)brugada pattern": "ST-T change",
    # 오타
}
type_map = {
    # 0 binary 1 discrete
    "dm": 0,
    "ekg": 0,
    "grwhich": 0,
    "code1": 1,
    # 2 continuous
    "age": 2,
    "albumin": 2,
    "alt": 2,
    "ast": 2,
    "bonet": 2,
    "bonez": 2,
    "bparmc": 2,
    "bmi": 2,
    "bun": 2,
    "creatine": 2,
    "crp": 2,
    "fef25": 2,
    "fev1": 2,
    "fvc": 2,
    "gripl1": 2,
    "gripr1": 2,
    "hb": 2,
    "hct": 2,
    "hdl": 2,
    "hba1c": 2,
    "glu0": 2,
    "height": 2,
    "hip": 2,
    "waist": 2,
    "labi": 2,
    "rabi": 2,
    "ldl": 2,
    "mch": 2,
    "mchc": 2,
    "mcv": 2,
    "muscle": 2,
    "plat": 2,
    "pulse": 2,
    "r_gtp": 2,
    "rbc": 2,
    "stiffness": 2,
    "t_bil": 2,
    "tchl": 2,
    "tg": 2,
    "u_ket": 2,
    "u_ph": 2,
    "uricacid": 2,
    "wbc": 2,
    "whr": 2,
    "weight": 2,
    "incell": 2,
    "excell": 2,
    "pbf": 2,
    "smam": 2,
    "smdu": 2,
    # 임의추가
    "skmm": 2,
    "axmm": 2,
    "armrm": 2,
    "armlm": 2,
    "legrm": 2,
    "leglm": 2,
    "asm": 2,
}
