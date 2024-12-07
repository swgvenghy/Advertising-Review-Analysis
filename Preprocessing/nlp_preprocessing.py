import os
import re
import json
import time
import requests
import numpy as np 
import pandas as pd
from konlpy.tag import Okt
from datetime import datetime
from add_word_in_OKT import add_word_in_Okt_dictionary as add_word

STOPWORDS_URL = "https://gist.githubusercontent.com/Nine1ll/3ec361da95ccd39101051ab57f86cf46/raw/a1a451421097fa9a93179cb1f1f0dc392f1f9da9/stopwords.txt"
FILE_DOWNLOAD_PATH = "Preprocessing/data/stopwords.txt"
DATA_PATH = "data/crawling/blog-review"

RESULT_PATH = "Preprocessing/result/"

# Version 1
# RESULT_PATH_ADD = "Preprocessing/result/result_add_words/"
# RESULT_PATH_NOUNS = "Preprocessing/result/result_nouns_test/"

# Version 2
# RESULT_PATH_POS = "Preprocessing/result/Pos/"
# RESULT_PATH_NOUNS = "Preprocessing/result/Nouns/"

# Version 3
RESULT_PATH_POS = "Preprocessing/result/integration_result/Pos/"
RESULT_PATH_NOUNS = "Preprocessing/result/integration_result/Nouns/"

RESTAURANT_NAME_PATH = "data/crawling/gangnamMatzip/gangnamOutput-modify.csv"
# FAKE_REVIEW_PATH = ""
seoul_districts = {
    "강남": "gangnam",
    "강동": "gangdong",
    "강북": "gangbuk",
    "강서": "gangseo",
    "관악": "gwanak",
    "광진": "gwangjin",
    "구로": "guro",
    "금천": "geumcheon",
    "노원": "nowon",
    "도봉": "dobong",
    "동대문": "dongdaemun",
    "동작": "dongjak",
    "마포": "mapo",
    "서대문": "seodaemun",
    "서초": "seocho",
    "성동": "seongdong",
    "성북": "seongbuk",
    "송파": "songpa",
    "양천": "yangcheon",
    "영등포": "yeongdeungpo",
    "용산": "yongsan",
    "은평": "eunpyeong",
    "종로": "jongno",
    "중랑": "jungnang",
    "중구": "junggu"
}


class review_preprocessing:
    """
    0. 사전 수정 예정 (키워드 추가) - 12/02 (완)
    1. 불용어 제외 (불용어 다운로드) - 11/28 (완)
    2. 조사 제외 - 11/29 (완)
    3. 가게 이름 추출 - 준호님이 해결 해주심. (스키마 변경으로 완)
    4. 유니코드 처리 불가 - navermap - 12/03(완)
    5. 혹시 모를 오류 대비 오류 처리 진행 중. (해당 파일만 추후 다시 작업 가능하도록.) - 12/07(완)
    6. Okt 사전 수정 코드.. - 12/07(완)
    7. 파일 스키마가 모두 다름... 왜 이렇게 주신 겁니까 선생님들.. ㅠ - 파일 분리로 해결..
    """
    def __init__(self, filepath="Preprocessing/data/stopwords.txt", github_url=None):
        self.okt = Okt()
        self.stopwords = self.load_stopwords(filepath, github_url)
        self.drop_pos = ["Josa", "Punctuation", "Foreign", "KoreanParticle"]
        self.district_dict = seoul_districts
        self.error_file_name = [] # error용 list


    def load_stopwords(self, filepath="Preprocessing/data/stopwords.txt", github_url=STOPWORDS_URL):
        stopwords = []
        if not os.path.exists(filepath):
            if github_url is None:
                raise ValueError("GitHub url이 제공되지 않았습니다.")

            # 다운로드하기
            response = requests.get(github_url)
            if response.status_code == 200:
                # 멀쩡하게 반응하면
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"stopwords.txt를 {github_url}에서 {filepath}에 다운로드 했습니다. ")
            else:
                raise Exception(f"우와 에러다. HTTP Status: {response.status_code} 실패했습니다.")

        # 파일 읽어오기
        with open(filepath, "r", encoding="utf-8") as f:
            stopwords = [line.strip() for line in f.readlines()]

        return stopwords


    def extract_hash_tags(self, text):
        """
        해시태그(#)가 포함된 단어를 추출하는 함수
        """
        text = str(text)
        hash_tags = re.findall(r"#\w+", text)  # 해시태그 추출
        return " ".join(tag[1:] for tag in hash_tags) if hash_tags else None 


    def take_res_name(self):
        fake_review_name = pd.read_csv(RESTAURANT_NAME_PATH)
        # print(fake_review_name.head(5))
        return fake_review_name["name"].to_list()


    def extract_res_name(self, title):
        """
        가게 이름 추출 함수
        """
        match = re.search(r'([가-힣]+(?:\s?[가-힣]+)*점)', str(title))  # '점'으로 끝나는 한글 단어 추출
        if match:
            print(match)
            return match.group(0)
        return None


    def data_raed(self):
        """
        리뷰 데이터가 있는 곳을 가져와서 데이터 이름 읽어오는 함수
        """
        review_data_path = []
        result_file_name = []
        file_names = os.listdir(DATA_PATH)
        for file_name in file_names:
            file_path = os.path.join(DATA_PATH, file_name)  # 전체 파일 경로
            if os.path.isfile(file_path):  # 파일이 있는 지 확인
                review_data_path.append(file_path) # 파일 위치
                result_file_name.append(os.path.splitext(os.path.basename(file_path))[0]) # 파일 이름용

        return review_data_path, result_file_name


    def preprocessing(self, text, norm=False, stem=False):
        text = str(text)
        text = self.regex_test(text)
        words = self.okt.pos(text,norm=norm, stem=stem)
        words = [word for word, pos in words if pos not in self.drop_pos]
        words = [self.regex(word) for word in words if word not in self.stopwords and self.regex(word).strip()]

        return " ".join(words)


    def preprocessing_nouns(self, text):
        text = str(text)
        text = self.regex_test(text)
        words = self.okt.nouns(text)
        words = [word for word in words if word not in self.stopwords and self.regex(word).strip()]

        return " ".join(words)


    def regex(self, text):
        """
        거슬리는 데이터들 삭제
        정규 표현식 패턴 (인터넷 주소와 "© NAVER Corp" 삭제)
        한 단어 이하 삭제
        공백 제거
        시간 형식 삭제
        """
        pattern = r"(© NAVER Corp|https?://\S+|www\.\S+|m\.\S+|\S+\.\S+\.\S+)"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?]', '', cleaned_text)
        cleaned_text = re.sub(r'[^\u0000-\uFFFF]', '', cleaned_text) # 이상한 기호 제거
        cleaned_text = re.sub(r'[\u1800-\u18AF]', '', cleaned_text) # 이상한 몽골 문자 제거
        cleaned_text = re.sub(r"\b\d{3,4}-\d{3,4}-\d{3,4}\b", "", cleaned_text)# 전화번호 형식 삭제 (예: 0507-1311, 010-1234-5678)
        cleaned_text = re.sub(r"\b\d+\b", "", cleaned_text)  # 숫자만 있는 부분을 제거
        cleaned_text = re.sub(r"\b\d{1,2}:\d{2}\b", "", cleaned_text) # 시간 형식 제외
        cleaned_text = re.sub(r"\b\w\b", "", cleaned_text) # 공백 제거
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip() # 한 단어 이하 삭제

        return cleaned_text


    def regex_test(self, text):
        """
        거슬리는 데이터들 삭제
        정규 표현식 패턴 (인터넷 주소와 "© NAVER Corp" 삭제)
        한 단어 이하 삭제
        공백 제거
        시간 형식 삭제
        """
        pattern = r"(© NAVER Corp|https?://\S+|www\.\S+|m\.\S+|\S+\.\S+\.\S+)"
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?]', '', cleaned_text)
        cleaned_text = re.sub(r'[^\u0000-\uFFFF]', '', cleaned_text) # 이상한 기호 제거
        cleaned_text = re.sub(r'[\u1800-\u18AF]', '', cleaned_text) # 이상한 몽골 문자 제거
        cleaned_text = re.sub(r"\b\d{3,4}-\d{3,4}-\d{3,4}\b", "", cleaned_text)# 전화번호 형식 삭제 (예: 0507-1311, 010-1234-5678)
        cleaned_text = re.sub(r"\b\d+\b", "", cleaned_text)  # 숫자만 있는 부분을 제거
        cleaned_text = re.sub(r"\b\d{1,2}:\d{2}\b", "", cleaned_text) # 시간 형식 제외
        cleaned_text = re.sub(r"\b\w\b", "", cleaned_text) # 공백 제거
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip() # 한 단어 이하 삭제

        return cleaned_text


    def convert_district_name(self, file_name):
        for kor, eng in self.district_dict.items():
            if kor in file_name:
                return file_name.replace(kor, eng)
        return file_name  # 변환이 필요 없는 경우 원본 파일 이름 반환


    def start_pos(self):
        print(f"Pos 추출을 실행합니다.")
        st = time.time()
        blog_reviews, reuslt_file_names = self.data_raed()
        for index, blog_review in enumerate(blog_reviews):
            start_time = time.time()
            print()
            print(blog_review)
            try:
                df = pd.read_csv(blog_review, encoding='utf-8-sig')
                df.dropna(inplace=True)
                if "가게 이름" in df.columns:
                    df.columns = ['Title', 'Content']

                df['hash_tag'] = df['Content'].apply(self.extract_hash_tags)
                df['title_pos'] = df['Title'].apply(lambda x: self.preprocessing(x, norm=True, stem=True))
                df['content_pos'] = df['Content'].apply(lambda x: self.preprocessing(x, norm=True, stem=True))
                
                df = df[['title_pos','content_pos','hash_tag']]
                reuslt_file_names[index] = self.convert_district_name(reuslt_file_names[index])

                df.to_csv(RESULT_PATH_POS+reuslt_file_names[index]+".csv")
            
            except Exception as e:
                self.error_file_name.append({blog_review: e})

            end_time = time.time()
            print(f"{reuslt_file_names[index]} 경과 시간: {(end_time - start_time):.2f} sec")
        
        et = time.time()
        print(f"Pos 추출을 종료.\n경과 시간: {(et-st):.2f}sec")


    def start_nouns(self):
        print(f"Nouns 추출을 실행합니다.")
        st = time.time()
        blog_reviews, reuslt_file_names = self.data_raed()
        for index, blog_review in enumerate(blog_reviews):
            
            start_time = time.time()
            print()
            print(blog_review)
            try:
                df = pd.read_csv(blog_review, encoding='utf-8-sig')
                df.dropna(inplace=True)
                if "가게 이름" in df.columns:
                    df.columns = ['Title', 'Content']

                df['hash_tag'] = df['Content'].apply(self.extract_hash_tags)
                df['title_pos'] = df['Title'].apply(lambda x: self.preprocessing_nouns(x))
                df['content_pos'] = df['Content'].apply(lambda x: self.preprocessing_nouns(x))
                
                df = df[['title_pos','content_pos','hash_tag']]
                reuslt_file_names[index] = self.convert_district_name(reuslt_file_names[index])
                
                df.to_csv(RESULT_PATH_NOUNS+reuslt_file_names[index]+".csv")
                
            except Exception as e:
                self.error_file_name.append({blog_review: e})

            end_time = time.time()
            print(f"{reuslt_file_names[index]} 경과 시간: {(end_time - start_time):.2f} sec")
        et = time.time()
        print(f"Nouns 추출을 종료.\n경과 시간: {(et-st):.2f}sec")


    def data_read_with_no_title(self):
        """
        review_data 스키마 변경으로 인한 새롭게 읽어오는 함수 
        """
        review_data_path = []
        result_file_name = []
        file_names = os.listdir("Preprocessing/row_review_data")
        for file_name in file_names:
            print(file_name)
            file_path = os.path.join("Preprocessing/row_review_data", file_name)  # 전체 파일 경로
            if os.path.isfile(file_path):  # 파일이 있는 지 확인
                review_data_path.append(file_path) # 파일 위치
                result_file_name.append(os.path.splitext(os.path.basename(file_path))[0]) # 파일 이름용

        return review_data_path, result_file_name


    def error_file_name_extract(self):
        # 혹시 error나온 파일이 있으면 그날에 에러난 파일을 txt file로 저장
        if len(self.error_file_name) > 0:
            today = datetime.now().strftime("%Y%m%d")
            file_name = f"error_file_{today}.txt"
            
            with open(file_name, "w") as file:
                for error in self.error_file_name:
                    file.write(error + "\n")  # 각 항목을 줄 단위로 저장

            print(f"파일 '{file_name}'에 저장되었습니다.")
        else:
            return None


    def start_pos_with_no_title(self):
        print(f"Pos 추출을 실행합니다.")
        st = time.time()
        blog_reviews, reuslt_file_names = self.data_read_with_no_title()

        for index, blog_review in enumerate(blog_reviews):
            start_time = time.time()
            print()
            print(blog_review)
            try:
                df = pd.read_csv(blog_review, encoding='utf-8',header=None)
                # df = pd.read_csv(blog_review, header=None)

                df.columns = ["res_name","Content"]
                # print(df.head(1))
                df.dropna(inplace=True)

                df['hash_tag'] = df['Content'].apply(self.extract_hash_tags)
                # df['res_name'] = df['Title'].apply(lambda x: self.preprocessing(x, norm=True, stem=True))
                df['content_pos'] = df['Content'].apply(lambda x: self.preprocessing(x, norm=True, stem=True))
                
                df = df[['res_name','content_pos','hash_tag']]
                reuslt_file_names[index] = self.convert_district_name(reuslt_file_names[index])
                df.to_csv(RESULT_PATH_POS+reuslt_file_names[index]+".csv")
            except Exception as e:
                self.error_file_name.append({blog_review: e})

            end_time = time.time()
            print(f"{reuslt_file_names[index]} 경과 시간: {(end_time - start_time):.2f} sec")
        
        et = time.time()
        print(f"Nouns 추출을 종료.\n경과 시간: {(et-st):.2f}sec")


    def start_nouns_with_no_title(self):
        print(f"Nouns 추출을 실행합니다.")
        st = time.time()
        blog_reviews, reuslt_file_names = self.data_read_with_no_title()
        for index, blog_review in enumerate(blog_reviews):
            start_time = time.time()
            print()
            print(blog_review)
            try:
                df = pd.read_csv(blog_review, encoding='utf-8',header=None)
                # df = pd.read_csv(blog_review,header=None)
                df.columns = ["res_name","Content"]
                # print(df.head(1))
                df.dropna(inplace=True)

                df['hash_tag'] = df['Content'].apply(self.extract_hash_tags)
                print("hash_tag 완료.")
                # df['res_name'] = df['Title'].apply(lambda x: self.preprocessing(x, norm=True, stem=True))
                df['content_pos'] = df['Content'].apply(lambda x: self.preprocessing_nouns(x))
                print("Content 완료.")
                df = df[['res_name','content_pos','hash_tag']]
                reuslt_file_names[index] = self.convert_district_name(reuslt_file_names[index])
                df.to_csv(RESULT_PATH_NOUNS+reuslt_file_names[index]+"_with_no_title.csv")

            except Exception as e:
                self.error_file_name.append({blog_review: e})
            
            end_time = time.time()
            print(f"{reuslt_file_names[index]} 경과 시간: {(end_time - start_time):.2f} sec")
        
        et = time.time()
        print(f"Pos 추출을 종료.\n경과 시간: {(et-st):.2f}sec")


if __name__ == "__main__":
    # 처음 실행 시 add_word 먼저 실행.
    # add_word.add_word_anaconda() # 처음 실행하면 주석을 풀고 실행해주세요.
    rp = review_preprocessing(filepath=FILE_DOWNLOAD_PATH, github_url=STOPWORDS_URL)
    rp.start_pos()
    rp.start_nouns()
    rp.start_pos_with_no_title()
    rp.start_nouns_with_no_title()
    rp.error_file_name_extract()
