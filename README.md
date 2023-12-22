# Pykoges

[Koges](https://nih.go.kr/ko/main/contents.do?menuNo=300566) (한국인유전체역학조사사업) 데이터를 읽어오기 위한 모듈입니다.

원주의과대학의 Koges-Arirang 데이터에 특화되어 있습니다.

```
pip install koges
```

```python
from koges import codingbook, koges
```

## koges.datatype

### `Question` 클래스
- 설문지 질문과 관련된 정보를 담은 클래스입니다.
- `add_answer` 메서드는 행 데이터에서 답변 정보를 추출하여 해당 질문 객체에 추가합니다.
- `add_fileinfo` 메서드는 파일 경로에서 데이터 타입과 연도를 추출하여 해당 질문 객체에 추가합니다.
- `from_row` 클래스 메서드는 행 데이터를 기반으로 질문 객체를 생성합니다.
- `to_json` 메서드는 질문 객체를 JSON 형식으로 변환하여 반환합니다.

### `Answer` 클래스
- 질문에 대한 답변을 담은 클래스입니다.
- `from_row` 클래스 메서드는 마지막 질문 객체와 행 데이터를 기반으로 답변 객체를 생성합니다.

### `Questions` 클래스
- 여러 질문 객체를 관리하는 클래스입니다.
- `keys` 메서드는 데이터 타입과 연도에 대한 키를 정렬하여 반환합니다.
- `from_type` 메서드는 특정 데이터 타입과 연도에 해당하는 질문 리스트를 반환합니다.
- `has_code` 메서드는 특정 코드를 가진 질문 리스트를 반환합니다.
- `has_text` 메서드는 특정 텍스트를 포함하는 질문 리스트를 반환합니다.

### `Patient` 클래스
- 환자 정보를 담은 클래스입니다.
- `to_json` 메서드는 환자 객체를 JSON 형식으로 변환하여 반환합니다.

### `Patients` 클래스
- 여러 환자 객체를 관리하는 클래스입니다.
- `append` 메서드는 새로운 환자 객체를 리스트에 추가합니다.
- `has_patient` 메서드는 특정 환자 객체가 리스트에 있는지 여부를 반환합니다.

## kogs.codingbook

### `read(folderName="./data_fixed")`
- 지정된 폴더에서 "codingbook"이 포함된 파일들을 찾아 `readCodingBook`을 호출하여 질문 데이터를 읽습니다.
- 결과로 `Questions` 클래스의 인스턴스를 반환합니다.

```python
q = codingbook.read("./data_fixed")
```

### `summary(q)`
- 분석 결과를 출력하는 함수입니다.
- Markdown 형식으로 출력되며, 실행 결과에는 전체 질문 개수, 코드 중복 제거 후 개수, 객관식과 주관식 데이터 개수, 연도별 질문 개수 등이 포함됩니다.

```python
q.summary()
```