# ADMT-RPA GitHub 업로드 가이드

## 준비된 파일들
✅ README.md
✅ .gitignore
✅ requirements.txt
✅ 모든 코드 파일

## GitHub에 업로드하는 방법

### 방법 1: GitHub Desktop 사용 (권장)

1. GitHub Desktop 설치 및 로그인
2. File → Add Local Repository
3. 경로: `C:\ai-agent\ADMT-RPA` 선택
4. "create a repository" 클릭
5. Repository name: `ADMT-RPA`
6. Description 입력 (선택)
7. Publish repository
   - ✅ Keep this code private 체크 해제 (공개 리포지토리)
   - Organization: minsk0429 선택
8. Publish!

### 방법 2: Git 명령어 사용

```bash
# 1. ADMT-RPA 폴더로 이동
cd C:\ai-agent\ADMT-RPA

# 2. Git 초기화
git init

# 3. 파일 추가
git add .

# 4. 커밋
git commit -m "Initial commit: ADMT with Revised Progressive Augmentation"

# 5. 원격 리포지토리 연결
git remote add origin https://github.com/minsk0429/ADMT-RPA.git

# 6. 푸시 (main 브랜치)
git branch -M main
git push -u origin main
```

### 방법 3: GitHub 웹사이트에서 직접 업로드

1. https://github.com/new 방문
2. Repository name: `ADMT-RPA`
3. Public 선택
4. "Create repository" 클릭
5. "uploading an existing file" 클릭
6. `C:\ai-agent\ADMT-RPA` 폴더의 모든 파일 드래그 앤 드롭
7. Commit 메시지 작성
8. "Commit changes" 클릭

## 업로드 전 체크리스트

✅ README.md - 수정 완료
✅ .gitignore - 생성 완료
✅ requirements.txt - 생성 완료
✅ code/ 폴더 및 모든 Python 파일
✅ cfgs/ 폴더 및 설정 파일
✅ train.py

## 리포지토리 URL
https://github.com/minsk0429/ADMT-RPA
