# Pykoges

[Koges](https://nih.go.kr/ko/main/contents.do?menuNo=300566) (한국인유전체역학조사사업) 데이터를 읽어오기 위한 모듈입니다.

원주의과대학의 Koges-Arirang 데이터에 특화되어 있습니다.

## 1. 클래스 및 함수 설명

### 1.1 `Question` 클래스
- 설문지 질문과 관련된 정보를 담은 클래스입니다.
- `add_answer` 메서드는 행 데이터에서 답변 정보를 추출하여 해당 질문 객체에 추가합니다.
- `add_fileinfo` 메서드는 파일 경로에서 데이터 타입과 연도를 추출하여 해당 질문 객체에 추가합니다.
- `from_row` 클래스 메서드는 행 데이터를 기반으로 질문 객체를 생성합니다.
- `to_json` 메서드는 질문 객체를 JSON 형식으로 변환하여 반환합니다.

### 1.2 `Answer` 클래스
- 질문에 대한 답변을 담은 클래스입니다.
- `from_row` 클래스 메서드는 마지막 질문 객체와 행 데이터를 기반으로 답변 객체를 생성합니다.

### 1.3 `Questions` 클래스
- 여러 질문 객체를 관리하는 클래스입니다.
- `keys` 메서드는 데이터 타입과 연도에 대한 키를 정렬하여 반환합니다.
- `from_type` 메서드는 특정 데이터 타입과 연도에 해당하는 질문 리스트를 반환합니다.
- `has_code` 메서드는 특정 코드를 가진 질문 리스트를 반환합니다.
- `has_text` 메서드는 특정 텍스트를 포함하는 질문 리스트를 반환합니다.

### 1.4 `Patient` 클래스
- 환자 정보를 담은 클래스입니다.
- `to_json` 메서드는 환자 객체를 JSON 형식으로 변환하여 반환합니다.

### 1.5 `Patients` 클래스
- 여러 환자 객체를 관리하는 클래스입니다.
- `append` 메서드는 새로운 환자 객체를 리스트에 추가합니다.
- `has_patient` 메서드는 특정 환자 객체가 리스트에 있는지 여부를 반환합니다.

## 2. 사용 예시

```python
# 사용 예시
questions_list = [...]  # Question 객체 리스트 생성
patients_list = [...]   # Patient 객체 리스트 생성

# Questions 클래스 인스턴스 생성
q = Questions(questions_list)

# Patients 클래스 인스턴스 생성
p = Patients(patients_list)

# 특정 데이터 타입과 연도에 해당하는 질문 객체 추출
selected_questions = q.from_type("baseline", 2022)

# 특정 코드를 가진 질문 객체 추출
specific_question = q.has_code("abc123")

# 특정 텍스트를 포함하는 질문 객체 추출
text_related_question = q.has_text("사용자 만족도 조사")

# 새로운 환자 객체를 리스트에 추가
new_patient = Patient({"name": "John Doe", "birthday": "19900101", "socialno1": "123456"})
p.append(new_patient)

# 특정 환자 객체가 리스트에 있는지 여부 확인
has_specific_patient = p.has_patient(new_patient)
```

## 1. codingbook 하위모듈

### 1.1 `readFromCsv(filePath)`
- 주어진 CSV 파일을 읽어서 openpyxl의 Workbook 형식으로 반환합니다.

### 1.2 `readCodingBook(filePath)`
- 주어진 CSV 파일에서 질문과 답변을 추출하여 Question 객체를 생성합니다.
- Question 객체는 `pykoges` 라이브러리의 `Question` 클래스의 인스턴스입니다.
- 파일 정보를 추가하고, 전체 질문 목록에 추가합니다.

### 1.3 `readCodingBooks(path="./data_fixed")`
- 지정된 폴더에서 "codingbook"이 포함된 파일들을 찾아 `readCodingBook`을 호출하여 질문 데이터를 읽습니다.
- 결과로 `Questions` 클래스의 인스턴스를 반환합니다.

### 1.4 `printInitResult(q)`
- 분석 결과를 출력하는 함수입니다.
- Markdown 형식으로 출력되며, 실행 결과에는 전체 질문 개수, 코드 중복 제거 후 개수, 객관식과 주관식 데이터 개수, 연도별 질문 개수 등이 포함됩니다.

## 2. 실행 결과 예시

```python
# 예시 데이터
q = readCodingBooks()
printInitResult(q)
```

