import os
import konlpy
import subprocess
import pandas as pd

# 절대 경로로 입력하셔야합니다.
KEWORD_PATH = "/Users/nine1ll/BDP_termProject/Hadoop-Classification/data/crawling/ad-keyword/ad_keyword_merged.csv"


class add_word_in_Okt_dictionary:
    """
    아나콘다 가상환경 기준으로 만든 함수 입니다.
    만약 로컬 환경의 사전을 변경하고 싶으면 konlpy.data.path에서 나오는 path의 마지막
    /data를 /java로 변경하시면 됩니다.
    """
    def __init__(self):
        self.path = self.get_path()
    

    def get_path(self):
        path = konlpy.data.path
        if len(path) > 1:
            print(f"python version에 맞는 Path를 선택해야합니다.\n{path}")
            for p in path:
                if "anaconda" in p:
                    return p[:-4]+"java"
        else:
            return path[0][:-4]+"java"
            # /opt/anaconda3/lib/python3.12/site-packages/konlpy/data
            # 여기서 data를 java로 바꿔야함

    def add_word_anaconda(self):
        if not self.path:
            print("올바른 Path를 설정하지 못했습니다. 종료합니다.")
            return  # Path가 없으면 실행 중단
        
        # 현재 작업 디렉토리를 변경
        try:
            os.chdir(self.path)
            print(f"현재 작업 디렉토리: {os.getcwd()}")
        except FileNotFoundError:
            print(f"경로를 찾을 수 없습니다: {self.path}")
            return
        
        try:
            result = subprocess.run(
                ["jar", "xvf", "open-korean-text-2.1.0.jar"],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)  # 명령 실행 결과 출력

            with open(f"{self.path}/org/openkoreantext/processor/util/noun/names.txt") as f:
                data = f.read()

            df = pd.read_csv(KEWORD_PATH)
            string = ""
            for word in df['keyword']:
                string += word+"\n"
            data += string
            
            with open(f"{self.path}/org/openkoreantext/processor/util/noun/names.txt", 'w') as f:
                f.write(data)
            
            try:
                result = subprocess.run(
                ["jar", "cvf", "open-korean-text-2.1.0.jar", 'org'],
                check=True,
                capture_output=True,
                text=True
                )
                print(result.stdout)  # 명령 실행 결과 출력
            except subprocess.CalledProcessError as e:
                print(f"명령 실행 중 오류 발생: {e.stderr}")
            
        except subprocess.CalledProcessError as e:
            print(f"명령 실행 중 오류 발생: {e.stderr}")
        except FileNotFoundError:
            print("jar 명령을 실행할 수 없습니다. Java가 설치되어 있는지 확인하세요.")

        