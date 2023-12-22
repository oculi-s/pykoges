class Question:
    def __init__(
        self,
        survey_name: str = None,
        survey_name_korean: str = None,
        survey_code: str = None,
        has_options: str = None,  # o (option), e (no option)
        variable_type: str = None,  # n (number),v (string)
        variable_length: int = 0,
        question_text: str = None,
        question_type: str = None,  # s (single),m (multi)
        answer: dict = None,
    ):
        self.survey_name = survey_name
        self.survey_name_korean = survey_name_korean
        self.survey_code = survey_code.lower()
        self.has_options = has_options
        self.variable_type = variable_type
        self.variable_length = variable_length
        self.question_text = question_text
        self.question_type = question_type
        self.answer = answer
        pass

    @staticmethod
    def __parsePath(path):
        import os

        name = os.path.split(path)[-1]
        name, _ = os.path.splitext(name)
        file_type, data_type, year = name.split("_")
        return [file_type, data_type, year]

    def add_answer(self, row):
        answer = Answer.from_row(self, row)
        self.answer[answer.code] = answer.text

    def add_fileinfo(self, filePath):
        file_type, data_type, year = self.__parsePath(filePath)
        self.data_type = data_type
        self.year = year

    @classmethod
    def from_row(cls, row):
        import inspect

        dim = len(inspect.signature(cls.__init__).parameters)
        row = [(x or "").strip() for x in row[:dim]]
        question = cls(*row[: dim - 2])
        question.answer = {}
        if question.question_type == "m" or row[8].isnumeric():
            question.add_answer(row)
        return question

    def to_json(self):
        import json

        data = self.__dict__.copy()
        return json.dumps(data, indent=4, ensure_ascii=False)


class Answer:
    def __init__(
        self,
        question: Question,
        code: str,
        test: str,
    ):
        self.question = question
        self.code = code
        self.text = test

    @classmethod
    def from_row(cls, last_question, row):
        question = last_question
        return cls(question, row[8], row[9])


class Questions:
    def __init__(
        self,
        lst=None,
        folderName="./data_fixed",
    ):
        import os

        self.list = lst if lst is not None else []
        self.len = len(self.list)
        self.folder = os.path.abspath(folderName)

    def keys(self, reverse=True, astype=list):
        import os

        def __sorting(a):
            if isinstance(a, str):
                a = str.split(a, " ")
            # data_type, year로 정렬
            return (a[0] == "track") * 100 + int(a[1])

        keys = []
        for d in set(self.__extract_attr(self, "data_type")):
            for y in set(self.__extract_attr(self, "year")):
                if not os.path.exists(f"{self.folder}/data_{d}_{y}.csv"):
                    continue
                if astype == list:
                    keys.append([d, y])
                else:
                    keys.append(f"{d} {y}")
        keys = list(sorted(keys, key=__sorting, reverse=reverse))
        return keys

    def append(self, x):
        self.list.append(x)

    @staticmethod
    def __filter(f, lst):
        return Questions(list(filter(f, lst)))

    @staticmethod
    def __find(f, lst):
        return next(filter(f, lst), None)

    @staticmethod
    def __valid_code(lst):
        res = []
        for x in lst:
            if hasattr(x, "survey_code"):
                new_code = "_".join(str.split(x.survey_code, "_")[1:])
                res.append(new_code)
        return list(set(res))

    def from_type(self, data_type, year):
        year = str(year).zfill(2)
        return self.__filter(
            lambda x: hasattr(x, "year")
            and hasattr(x, "data_type")
            and year == x.year
            and data_type == x.data_type,
            self.list,
        )

    def has_code(self, code):
        if isinstance(code, list) or isinstance(code, set):
            res = Questions()
            for x in code:
                f = lambda y: y.survey_code.endswith(f"_{x.lower()}")
                q = self.__find(f, self.list)
                res.append(q)
            return res
        return self.__filter(
            lambda x: x.survey_code.endswith(f"_{code.lower()}"),
            self.list,
        )

    def has_text(self, code):
        if isinstance(code, list):
            res = Questions()
            for x in code:
                q = self.__find(lambda y: x in y.question_text, self.list)
                res.append(q)
            return res
        return self.__filter(lambda x: code in x.question_text, self.list)

    @staticmethod
    def __extract_attr(self, name):
        return [getattr(x, name) if hasattr(x, name) else None for x in self.list]

    def __getattr__(self, name):
        if name not in self.__dict__:
            self.__dict__[name] = self.__extract_attr(self, name)
        return self.__dict__[name]

    def summary(self):
        from IPython.display import Markdown, display

        year_list = sorted(set(self.year))
        valid_code = self.__valid_code(self.list)

        res = "#### 실행결과\n"
        res += "***\n"
        res += "#### 1. 전체 질문데이터\n"
        res += "***\n"
        res += f"- 전체 질문 데이터 **{self.len}**개\n"
        res += f"- 객관식 데이터 **{len([x for x in self.list if x.answer])}**개 / 주관식 데이터 **{len([x for x in self.list if not x.answer])}**개\n"
        res += f"- 코드 중복 제거시 **{len(valid_code)}**개\n"
        res += f"- 연도별 질문 개수\n\n"

        res += f"||{'|'.join(year_list)}|\n"
        res += f"|:-:|{':-:|'*len(year_list)}\n"
        res += f"|baseline 질문 수|{'|'.join([str(self.from_type('baseline', year).len) for year in year_list])}|\n"
        res += f"|track 질문 수|{'|'.join([str(self.from_type('track', year).len) for year in year_list])}|\n"

        res += "***\n"
        res += "#### 3. 예시 데이터\n"
        res += "***\n"
        res += "- 질문 데이터\n"
        res += "```json\n"
        res += f"{self.__find(lambda x:len(x.answer)>3, self.list).to_json()}"
        res += "```\n"

        display(Markdown(res))


class Patient:
    def __init__(self, json):
        if json["socialno2"]:
            if json["socialno2"] != json["socialno2"] or "*" in json["socialno2"]:
                json["socialno2"] = None
        for k, v in json.items():
            setattr(self, k, v)

    def __eq__(self, other):
        if not isinstance(other, Patient):
            return False
        if self.cp and self.cp == other.cp:
            return True
        if self.socialno2 and self.socialno2 == other.socialno2:
            return True
        if self.name and self.name == other.name:
            if self.socialno1 and self.socialno1 == other.socialno1:
                return True
            if self.birthday and self.birthday == other.birthday:
                return True
        return False

    def __getattr__(self, name):
        if name not in self.__dict__:
            return None
        return self.__dict__[name]

    def to_json(self):
        return {
            "name": self.name,
            "birthday": self.birthday,
            "socialno1": self.socialno1,
            "socialno2": self.socialno2,
        }


class Patients:
    def __init__(self, list):
        self.list = list

    def append(self, p):
        self.list.append(p)

    def has_patient(self, patient):
        for p in self.list:
            if p == patient:
                return True
        return False
