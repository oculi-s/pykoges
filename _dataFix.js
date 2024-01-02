/**
 * 2023.12.04 dataFix 파일
 * 
 * python의 openpyxl이나 pandas로 xlsx파일을 읽어오면 로딩시간이 너무 오래 걸려 시간을 줄이고  
 * 데이터간의 코드 / 데이터 / 양식의 통합이 되지 않은 상황이 많아 이를 통합하고자  
 * nodejs라는 언어를 이용해 xlsx파일을 csv파일로 변환하고 데이터를 통합하는 함수입니다.  
 * 
 * (https://nodejs.org/)로 접속하셔서 nodejs를 설치한 뒤 이 파일을 실행하면  
 * 현재 data폴더 내에 있는 파일들이 전부 data_fixed 폴더로 변환되어 저장되나  
 * 이미 변환된 파일을 드라이브에 업로드 해두었으니 실행시키지 않아도 됩니다.  
 */

const fs = require('fs');
const path = require('path');
const xlsx = require('xlsx');

const d = path.join(__dirname, './data'); // 아리랑 코호트
const d2 = path.join(__dirname, './data', '2020_2022'); // 인지노화 코호트
const df = path.join(__dirname, './data_fixed'); // 변환된 파일이 저장될 문서
if (!fs.existsSync(df)) fs.mkdirSync(df, { recursive: true });

// 코드 임의수정
let _varfixmap = require('./_varfixmap')
let _inbodyMap = require('./_inbodymap');

String.prototype.splitCSV = function () {
    var matches = this.match(/(\s*"[^"]+"\s*|\s*[^,]+|,)(?=,|$)/g);
    for (var i = 0; i < matches?.length; ++i) {
        matches[i] = matches[i].trim();
        if (matches[i] == ',') matches[i] = '';
    }
    if (this[0] == ',') matches.unshift("");
    return matches;
}
Array.prototype.transpose = function () {
    return this[0].map((col, i) => this.map(row => row[i]));
}

async function fix() {
    // 파일 이름으로부터 file_type, data_type, from, to를 가져오는 함수
    // file_type = codingbook / data
    // data_type = baseline / track
    function parsePath(e) {
        const name = path.parse(e).name;
        let year = name.split('(')[1].replace(')', '');
        let from = year.slice(2, 4), to = year.slice(7, 9);
        let res = {};
        if (from) res.from = parseInt(from);
        if (to) res.to = parseInt(to);
        res.file_type = e.includes('코딩북') ? 'codingbook' : 'data';
        res.data_type = e.includes('기반') ? 'baseline' : 'track';
        if (e.includes('인지노화')) {
            if (res.from == 20 && res.file_type == 'codingbook') res.to = 22
        } else {
            if (res.from == 9 && res.data_type == 'track') res.to = 10
        }
        return res;
    }
    // file_type, data_type, year로부터 저장할 파일 이름을 가져오는 함수
    function makeName({ file_type, data_type, year }) {
        year = String(year).padStart(2, '0')
        return `${file_type}_${data_type}_${year}.csv`
    }
    // decode range, 엑셀 시트의 전체 범위를 가져오는 함수
    const dr = (ws) => xlsx.utils.decode_range(ws["!ref"]);
    // encode range, 엑셀 시트의 범위를 xlsx 모듈이 인식할 수 있도록 변환하는 함수
    const er = (ref) => xlsx.utils.encode_range(ref.s, ref.e);
    // decode cell, 셀위치 (ex. A1)를 좌표로 변환하는 함수
    const dc = (cell) => xlsx.utils.decode_cell(cell);
    // encode cell, 좌표를 셀위치 (ex. A1)로 변환하는 함수
    const ec = (r, c) => xlsx.utils.encode_cell({ r, c });
    // from to, 현재 파일의 처음 셀 좌표부터 끝 좌표 셀을 가져오는 함수
    function ft(ws) {
        const ref = ws['!ref'].split(':');
        if (ref.length == 1)
            return [ref[0], ref[0]].map(dc);
        return ref.map(dc);
    }
    // 처음으로 값이 존재하는 셀을 가져오는 함수
    function firstCell(ws) {
        var ref = dr(ws);
        for (let c = ref.s.c; c <= ref.e.c; c++) {
            for (let r = ref.s.r; r <= ref.e.r; r++) {
                const cell = ec(r, c);
                if (ws[cell]) return cell;
            }
        }
        return null;
    }
    // 행삭제, python의 openpyxl과 달리 내장함수가 없어 구현해야 합니다.
    async function deleteRow(ws, row) {
        var ref = dr(ws);
        for await (var r of range(row, ref.e.r)) {
            for await (var c of range(ref.s.c, ref.e.c + 1)) {
                ws[ec(r, c)] = ws[ec(r + 1, c)];
                delete ws[ec(r + 1, c)];
            }
        }
        ref.e.r--
        ws['!ref'] = er(ref);
    }
    // 열삭제, python의 openpyxl과 달리 내장함수가 없어 구현해야 합니다.
    async function deleteCol(ws, col) {
        var ref = dr(ws);
        for await (var c of range(col, ref.e.c)) {
            for await (var r of range(ref.s.r, ref.e.r + 1)) {
                ws[ec(r, c)] = ws[ec(r, c + 1)];
                delete ws[ec(r, c + 1)];
            }
        }
        ref.e.c--
        ws['!ref'] = er(ref);
    }
    // 이름을 기준으로 행삭제
    async function deleteColName(ws, val) {
        var ref = dr(ws);
        for await (var c of range(ref.s.c, ref.e.c)) {
            if (ws[ec(ref.s.r, c)]?.v?.includes(val)) {
                deleteCol(ws, c);
            }
        }
    }
    // python의 np.arange와 같은 함수
    function range(start, end) {
        let array = [];
        for (let i = start; i < end; ++i) {
            array.push(i);
        }
        return array;
    }
    // 셀 내에 존재하는 줄바꿈을 제거합니다.
    async function fixCodingBook(ws) {
        for await (let cell of Object.keys(ws)) {
            if (cell.startsWith('!')) continue;
            if (ws[cell]?.r?.constructor != String) continue;
            ws[cell].r = ws[cell]?.r?.replace(/\r\n|\n|"-"|/g, '');
            ws[cell].v = ws[cell]?.v?.replace(/\r\n|\n|"-"|/g, '');
            ws[cell].w = ws[cell]?.w?.replace(/\r\n|\n|"-"|/g, '');
        }
    }
    // 셀 내에 존재하는 결측치를 제거합니다.
    async function fixData(ws) {
        for await (let cell of Object.keys(ws)) {
            if (cell.startsWith('!')) continue;
            if (ws[cell]?.r?.constructor != String) continue;
            // 0.9*과 같은 결측치 수정
            ws[cell].r = ws[cell]?.r?.replace(/\*/g, '');
            ws[cell].v = ws[cell]?.v?.replace(/\*/g, '');
            ws[cell].w = ws[cell]?.w?.replace(/\*/g, '');
            // 1..7과 같은 데이터 수정
            ws[cell].r = ws[cell]?.r?.replace(/\.\./g, '.');
            ws[cell].v = ws[cell]?.v?.replace(/\.\./g, '.');
            ws[cell].w = ws[cell]?.w?.replace(/\.\./g, '.');
        }
    }
    // 셀 병합이 있는 경우 모든 병합 셀 내에 같은 값을 채워줍니다.
    async function fillMerges(ws) {
        if (!ws['!merges']?.length) return;
        for await (let merge of ws["!merges"]) {
            let cols = range(merge.s.c, merge.e.c + 1);
            let rows = range(merge.s.r, merge.e.r + 1);
            for await (let c of cols) {
                for await (let r of rows) {
                    xlsx.utils.sheet_add_aoa(
                        ws,
                        [[ws[xlsx.utils.encode_col(merge.s.c) + xlsx.utils.encode_row(merge.s.r)]?.v]],
                        { origin: xlsx.utils.encode_col(c) + xlsx.utils.encode_row(r) }
                    )
                }
            }
        }
    }

    // 20-22 데이터의 inbody의 코드가 한글이며
    // 관련 코드가 코딩북에 정리되어 있지 않아
    // 한글을 영어 코드로 변환하고 코딩북에 추가하기 위한 데이터입니다.
    let addInbodyToCodingBook = {}

    // 데이터를 읽어옵니다.
    async function readData(filePath) {
        if (!fs.existsSync(filePath)) return '';
        let all = '';
        const data = xlsx.readFile(filePath);
        // 코딩북은 모든 시트의 내용을 하나로 합쳐줍니다.
        // 데이터는 맨 처음 시트의 내용만 가져옵니다.
        // (나머지 시트는 다른 연구자 분들이 임의로 수정한 내용입니다.)
        for await (let sheetName of data.SheetNames) {
            let ws = data.Sheets[sheetName];
            ws['!ref'] = er({ s: { r: 0, c: 0 }, e: dr(ws).e });
            if (filePath.includes('코딩북')) {
                // 코딩북인 경우
                await fixCodingBook(ws);
                await fillMerges(ws);

                let csvData;
                // 05-07 코딩북은 수정할 부분이 많습니다.
                if (filePath.includes('05')) {
                    csvData = xlsx.utils.sheet_to_csv(ws);
                    // 데이터 분류를 적어놓은 행을 삭제합니다.
                    csvData = csvData.split('\n')
                        ?.map(e => e.trim())
                        ?.map(e => e.splitCSV())
                        ?.filter(e => e[1])
                        ?.filter(e => !e[0].startsWith('번호')) // 첫행 형태 지우기
                    let newCsvData = [];
                    for await (let e of csvData) {
                        let survey_code = e[1];
                        let question_text = e[4];
                        let variable_type = e[5];
                        let variable_length = e[6];
                        let a = e[7];

                        let has_options = 'e';
                        let question_type = has_options == 'o' ? 'm' : 's';
                        let answers = '';
                        e = [
                            '', // survey name
                            '', // survey name korean
                            survey_code,
                            has_options,
                            variable_type,
                            variable_length,
                            question_text,
                            question_type,
                            answers
                        ]
                        // 질문 선지를 기록한 방식이 다른 코딩북과 다릅니다.
                        // 객관식은 선지를 나누어줍니다.
                        if (/^(\"?\d+[:=\.])/.test(a)) {
                            a = a
                                ?.replace(/"/g, '')
                                ?.split(/(\d+[:=\.])/)
                                ?.map(e => e.trim().replace(/,|\.|=|:/g, '').trim())
                                ?.filter(e => e)
                            // answer를 [번호, 선지]로 나누어주었기 때문에
                            // 홀수개가 나오면 나누는 과정에서 오류가 있는 것입니다.
                            if (a.length % 2) console.log(a)
                            e[3] = 'o';
                            // 처음 선지를 가져옵니다.
                            e[8] = a[0], e[9] = a[1];
                            newCsvData.push(e.join(','))
                            // 나머지 선지를 가져옵니다.
                            for await (let i of range(1, a.length / 2)) {
                                e = ['', '', '', '', '', '', '', '', a[i * 2], a[i * 2 + 1]];
                                newCsvData.push(e.join(','))
                            }
                        } else {
                            // 주관식은 그냥 추가해줍니다.
                            newCsvData.push(e.join(','))
                        }
                    }
                    csvData = newCsvData.join('\n');
                } else {
                    // 처음 셀이 시작셀이 될때까지 첫행과 첫열을 지워줍니다.
                    while (
                        !String(ws[firstCell(ws)]?.v)?.startsWith('설문지명') &&
                        !String(ws[firstCell(ws)]?.v)?.startsWith('베타테스트') && // 14추적, 혈액검사
                        ft(ws)[0].c < ft(ws)[1].c
                    ) await deleteCol(ws, 0);
                    while (
                        !String(ws[firstCell(ws)]?.v)?.startsWith('설문지명') &&
                        !String(ws[firstCell(ws)]?.v)?.startsWith('베타테스트') && // 14추적, 혈액검사
                        ft(ws)[0].r < ft(ws)[1].r
                    ) await deleteRow(ws, 0);

                    // 전체 sheet범위의 마지막 셀의 열을 11로 맞춰줍니다.
                    // 코딩북에서 비고 열이나 코딩규칙 열을 삭제하는 과정입니다.
                    while (ft(ws)[1].c >= 11) { await deleteCol(ws, 11); }

                    // 삭제하고 난 뒤 유효한 데이터가 없는 경우 비워줍니다.
                    if (
                        !String(ws[firstCell(ws)]?.v)?.startsWith('설문지명') &&
                        !ws['!ref'].includes(':')
                    ) ws = {};

                    csvData = xlsx.utils.sheet_to_csv(ws);
                }
                csvData = csvData.split('\n')
                    .filter(e => e.splitCSV())
                    ?.map(e => {
                        e = e.splitCSV();
                        // 결측치 처리과정입니다.
                        if (!e[2]) e[0] = e[1] = ''; // 질문선지에도 설문 코드를 입력해둔 경우
                        if (e[4] == '1' || e[4] == 'c') e[4] = 'v'; // 문자열을 1이나 c로 설정한 경우
                        if (e[3] == '0' || e[3] == 'O') e[3] = 'o'; // 옵션여부를 0이나 O로 설정한 경우
                        if (e[3] == 'A' || e[3] == 'n') e[3] = 'e'; // 옵션여부를 A이나 n으로 설정한 경우
                        if (e[2].endsWith('08')) e[2] = e[2].slice(0, -2); // 질문 코드를 잘못 설정한 경우
                        e[2] = e[2].toLowerCase();
                        if (e[2].includes('_')) {
                            let prefix = e[2].split('_')[0]
                            let code = e[2].split('_').slice(1).join('_')
                            code = _varfixmap[code] || code
                            e[2] = `${prefix}_${code}`
                        }
                        return e.join(',');
                    })?.filter(e => {
                        e = e.splitCSV();
                        if (e[0].startsWith('설문지명')) return false; // 첫행 형태 지우기
                        if (e[8] == '코드') return false; // 두번째 행 형태 지우기
                        return true;
                    })?.join('\n');
                all += '\n' + csvData;
            } else {
                // 2013데이터의 결측치를 처리합니다.
                if (filePath.includes('13')) {
                    await fixData(ws);
                }
                // let remVars = ['_insp', '_createdtm', '_relative', '_relname', '_relp']
                // for await (let v of remVars) {
                //     await deleteColName(ws, v);
                // }
                // 데이터인 경우
                let csvData = xlsx.utils.sheet_to_csv(ws);
                csvData = csvData.split('\n');
                let prefix = csvData[0].splitCSV()[0]?.split('_')[0].toLowerCase(); // NA1과 같은 코드 접두사
                // 인지노화 데이터는 ekg와 inbody데이터를 통합해줍니다.
                if (
                    filePath.includes('인지노화') &&
                    !filePath.includes('_ekg') &&
                    !filePath.includes('_inbody')
                ) {
                    let { from } = parsePath(filePath);
                    let { dir, name } = path.parse(filePath);
                    // 재귀적으로 데이터를 읽어옵니다.
                    let csvEkg = await readData(path.join(dir, `${name}_ekg.xlsx`));
                    let csvInbody = await readData(path.join(dir, `${name}_inbody.xlsx`));
                    // array 형태로 만들어줍니다.
                    csvEkg = csvEkg?.split('\n')?.map(e => e.splitCSV())?.filter(e => e);
                    csvInbody = csvInbody?.split('\n')?.map(e => e.splitCSV())?.filter(e => e);

                    // 인지노화는 inbody 코드가 한글로 되어 있어 수정해야 합니다.
                    // inbodycode 는 기존의 아리랑 코드와 인지노화 inbody 코드를 미리 매칭시켜둔 데이터입니다.
                    // 기존에 없던 코드를 임의로 추가한 경우 (ex. _neck) 가 있습니다.
                    if (csvInbody.length) csvInbody[0] = csvInbody[0]
                        ?.map(e => e.split(',')?.slice(1)?.join('').replace(/ |"/g, '') // 2020은 쉼표가 포함
                            || e.split('.')?.slice(1)?.map(e => e.trim())?.join('')) // 2021,2022는 .을 포함
                        ?.map(e => {
                            if (_inbodyMap[e] != '' && _inbodyMap[e] != undefined) {
                                let ne = `${prefix}_${_inbodyMap[e]}`;
                                // 만약 기존의 코드가 존재한다면 바꾸지 않습니다.
                                if (csvData[0]?.splitCSV()?.find(e => e == ne)) {
                                    return e;
                                }
                                // inbody 데이터는 코딩북에 존재하지 않으므로
                                // 만약 미리 정해진 데이터가 있다면 코딩북에 추가해줘야 합니다.
                                // 연도별로 추가할 변수를 정해줍니다.
                                if (addInbodyToCodingBook[from] == undefined) {
                                    addInbodyToCodingBook[from] = {};
                                }
                                // 한글 값과 영어 코드를 저장해준 뒤, 모든 파일을 변환한 후에 20-22코딩북을 수정하겠습니다.
                                addInbodyToCodingBook[from][e] = ne;
                                return ne;
                            }
                            return e;
                        });

                    csvData = csvData
                        ?.map(e => e.splitCSV())
                        ?.map((e, i) => {
                            if (i == 0) {
                                // 첫행에 ekg와 inbody 개별 데이터의 질문 코드를 추가합니다.
                                if (csvEkg.length) e.push(...csvEkg[0]);
                                if (csvInbody.length) e.push(...csvInbody[0]);
                                e = e.map(c => c.toLowerCase())
                            } else {
                                // 환자 id가 같은 행을 찾아줍니다.
                                if (csvEkg.length) {
                                    let ekgRow = csvEkg.find(row => row[0] == e[0]);
                                    // 존재하는 경우 그냥 추가, 아닌 경우 행의 길이만큼의 빈 데이터를 추가합니다.
                                    if (ekgRow) e.push(...ekgRow);
                                    else e.push(...Array(csvEkg[0].length).fill(''));
                                }
                                if (csvInbody.length) {
                                    let inbodyRow = csvInbody.find(row => row[0] == e[0]);
                                    if (inbodyRow) e.push(...inbodyRow);
                                    else e.push(...Array(csvInbody[0].length).fill(''));
                                }
                            }
                            return e;
                        })

                    csvData = csvData
                        ?.transpose()
                        ?.filter(e => !e[0].includes('비공개')) // 코드가 비공개된 경우 삭제
                        ?.filter((e, i) => csvData[0].indexOf(e[0]) == i) // 열의 데이터가 겹치는 경우 삭제
                        ?.transpose();

                    // 모든 행의 열의 개수가 같지 않은 경우 오류체크
                    let sizeCheck = (new Set(csvData.map(e => e.length))).size == 1;
                    if (!sizeCheck) console.error(`인지노화 데이터 통합에서 오류가 발생했습니다.\nfilepath : ${filePath}`);
                    csvData = csvData.map(e => e.join(','))
                }
                // 데이터는 첫 행의 질문 코드가 잘못된 경우만 수정합니다.
                csvData[0] = csvData[0]
                    ?.splitCSV()
                    ?.map(e => {
                        e = e.toLowerCase();
                        if (e.endsWith('08')) e = e.slice(0, -2); // 질문 코드를 잘못 설정한 경우
                        if (e.includes('_')) {
                            let code = e.split('_').slice(1).join('_')
                            code = _varfixmap[code] || code
                            e = `${prefix}_${code}`
                        } else if (_varfixmap[e]) {
                            e = `${prefix}_${_varfixmap[e]}`;
                        }
                        return e;
                    })?.join(',');
                return csvData.join('\n');
            }
        }
        all = all
            .split('\n')
            ?.filter(e => e.split(',').filter(e => e).length) // 빈 행 제거
            ?.join('\n')
        return all;
    }

    // 새로 만들어야 하는지 여부를 정하는 함수
    function shouldCreate(e) {
        // 임의 수정 데이터인경우
        if (!e.includes('기반') &&
            !e.includes('추적') &&
            !e.includes('코딩북') &&
            !e.includes('인지노화')) return false;
        // 인지노화 개별데이터인 경우
        if (e.includes('_ekg') || e.includes('_inbody')) return false;
        // 엑셀이 아닌 경우, 엑셀 임시파일인 경우
        if (!e.endsWith('xlsx') || e.startsWith('~')) return false;

        // if (e.includes('코딩북')) return true;
        return true;
        var { data_type, file_type, from, to } = parsePath(e);
        if (to) {
            // 여러 해에 걸친 데이터인 경우 하나라도 없으면 새로 만듬
            for (let y = from; y <= to; y++) {
                let name = makeName({ file_type, data_type, year: y });
                let csvFilePath = path.join(df, name);
                if (!fs.existsSync(csvFilePath)) return true;
            }
            return false;
        } else {
            let name = makeName({ file_type, data_type, year: from });
            let csvFilePath = path.join(df, name);
            return !fs.existsSync(csvFilePath);
        }
    }
    // 05-17 아리랑 데이터 수정
    for await (let e of fs.readdirSync(d)) {
        const filePath = path.join(d, e);
        if (fs.lstatSync(filePath).isDirectory()) continue;
        const { data_type, file_type, from, to } = parsePath(e);
        if (!shouldCreate(e)) continue;
        let data = await readData(filePath);
        if (to) {
            for (let y = from; y <= to; y++) {
                let name = makeName({ file_type, data_type, year: y });
                console.log(name);
                let csvFilePath = path.join(df, name);
                if (from == 5 && !e.includes('코딩북')) {
                    // 05-07 데이터는 통합되어있어 방문 일자를 바탕으로 나누어줍니다.
                    let data57 = data.split('\n')
                    data57 = [data57[0], ...data57
                        .filter(e => e
                            .splitCSV()
                            ?.slice(-8)[0]
                            ?.startsWith(`20${String(y).padStart(2, '0')}`)
                        )].join('\n')
                    fs.writeFileSync(csvFilePath, data57, 'utf8');
                } else {
                    fs.writeFileSync(csvFilePath, data, 'utf8');
                }
            }
        } else {
            let name = makeName({ file_type, data_type, year: from });
            let csvFilePath = path.join(df, name);
            console.log(name);
            fs.writeFileSync(csvFilePath, data, 'utf8');
        }
    };

    // 20-22 인지노화 데이터 수정
    for await (let e of fs.readdirSync(d2)) {
        const filePath = path.join(d2, e);
        if (fs.lstatSync(filePath).isDirectory()) continue;
        const { data_type, file_type, from, to } = parsePath(e);
        if (!shouldCreate(e)) continue;
        let data = await readData(filePath);
        if (to) {
            for (let y = from; y <= to; y++) {
                let name = makeName({ file_type, data_type, year: y });
                console.log(name);
                let csvFilePath = path.join(df, name);
                fs.writeFileSync(csvFilePath, data, 'utf8');
            }
        } else {
            let name = makeName({ file_type, data_type, year: from });
            let csvFilePath = path.join(df, name);
            console.log(name);
            fs.writeFileSync(csvFilePath, data, 'utf8');
        }
    };

    // 20-22 인바디 데이터 중 코딩북에 추가되지 않은 데이터를 추가하는 코드
    for await (let year of Object.keys(addInbodyToCodingBook)) {
        let name = makeName({ file_type: 'codingbook', data_type: 'track', year });
        let dir = path.join(df, name);
        let codingbook = fs.readFileSync(dir, { encoding: 'utf-8' }).split('\n');
        for await (let question_text of Object.keys(addInbodyToCodingBook[year])) {
            let survey_code = addInbodyToCodingBook[year][question_text];
            let row = [
                'Inbody', '인바디임의추가',
                survey_code,
                'e', // 주관식
                'n', // 숫자
                0, // 변수길이 미설정
                question_text,
                '', '', '', '' // 나머지는 비워줌
            ]
            if (!codingbook.find(e => e.splitCSV()[2] == survey_code)) {
                codingbook.push(row.join(','));
            }
        }
        fs.writeFileSync(dir, codingbook.join('\n'))
    }
}

fix();