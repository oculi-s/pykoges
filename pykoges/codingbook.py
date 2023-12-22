@staticmethod
def __readCodingBook(filePath):
    from pykoges.datatype import Question
    import csv, openpyxl

    questions_list = []
    # 파일읽기
    wb = openpyxl.Workbook()
    db = wb.active
    with open(filePath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            db.append(row)

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


def read(folderName):
    from pykoges.datatype import Questions
    from tqdm.notebook import tqdm
    import os

    folder = os.path.abspath(folderName)
    if not os.path.exists(folder):
        raise FileExistsError("파일을 읽어올 경로가 존재하지 않습니다.\n폴더 이름을 다시 설정해주세요.")
    # 중복실행을 대비해 초기 변수들을 비워줍니다.
    questions_list = []
    # 확장자가 없거나 (폴더)
    # 엑셀을 실행시켰을 때 생기는 임시파일 (~$...)인경우 통과
    files = filter(
        lambda x: os.path.splitext(x)[1]
        and not x.startswith("~")
        and "codingbook" in os.path.splitext(x)[0],
        os.listdir(folder),
    )
    for x in tqdm(list(files), desc="코딩북 읽어오는중..."):
        # 파일 확장자 분리
        name, ext = os.path.splitext(x)
        filePath = os.path.join(folder, x)
        # 코딩북인경우 readCodingBook실행
        if "codingbook" in name:
            questions_list += __readCodingBook(filePath)
    return Questions(questions_list, folderName=folderName)
