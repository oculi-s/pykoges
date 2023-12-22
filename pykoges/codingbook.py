import csv, os, openpyxl
from pykoges.data import Question, Questions
from IPython.display import display, Markdown


def readFromCsv(filePath):
    wb = openpyxl.Workbook()
    db = wb.active
    with open(filePath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            db.append(row)
    return db


def readCodingBook(filePath):
    questions_list = []
    # 파일읽기
    db = readFromCsv(filePath)
    question = None
    # 열의 개수가 1이상이고 두번째 행, 첫번째 열의 데이터가 존재하는 경우
    if db.max_column > 0 and db.max_row > 0:
        for row in db.iter_rows(2):
            # openpyxl 라이브러리는 cell.value를 통해 값을 호출하므로 값의 리스트를 가져오도록 설정
            row = [x.value for x in row]
            # 첫행인 경우 통과
            if str.startswith(row[0], "설문지명"):
                continue
            # 행의 8개 데이터중 어떤 것이라도 있는 경우 (질문 정보)
            elif any(row[:8]):
                # 행 데이터를 바탕으로 질문 생성
                question = Question.from_row(row)
                # 파일 정보 추가
                question.add_fileinfo(filePath)
                # 전체 질문 목록에 추가
                questions_list.append(question)
            # 행의 8개 데이터가 모두 빈 경우 경우 (질문 선지)
            elif question:
                # 설정된 질문에 답변 추가
                question.add_answer(row)
    return questions_list


def readCodingBooks(path="./data_fixed"):
    # 중복실행을 대비해 초기 변수들을 비워줍니다.
    questions_list = []
    # 'data_fixed' 폴더에 있는 데이터를 로드
    for x in os.listdir(path):
        # 파일 확장자 분리
        name, ext = os.path.splitext(x)

        # 확장자가 없거나 (폴더)
        # 엑셀을 실행시켰을 때 생기는 임시파일 (~$...)인경우 통과
        if not ext or "~$" in name:
            continue

        filePath = os.path.join(path, x)
        # 코딩북인경우 readCodingBook실행
        if "codingbook" in name:
            questions_list += readCodingBook(filePath)
    return Questions(questions_list)


def printInitResult(q):
    res = "#### 실행결과  "
    year_list = sorted(set(q.year))
    res += f"""
***
#### 1. 전체 질문데이터
***
- 전체 질문 데이터 **{len(q.list)}**개
- 코드 중복 제거시 **{len(set(q.valid_code))}**개
- 객관식 데이터 **{len([x for x in q.list if x.answer])}**개 / 주관식 데이터 **{len([x for x in q.list if not x.answer])}**개
- 연도별 질문 개수

"""
    res += f"""
||{'|'.join(year_list)}|  
|:-:|{':-:|'*len(year_list)}  
|baseline 질문 수|{'|'.join([str(q.from_type('baseline', year).len) for year in year_list])}|  
|track 질문 수|{'|'.join([str(q.from_type('track', year).len) for year in year_list])}|  
"""
    res += f"""
***
#### 3. 예시 데이터
***
- 질문 데이터
```json
{q.list[8].to_json()}
```
"""
    display(Markdown(res))
