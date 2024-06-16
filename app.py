import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import random
import collections
collections.Callable = collections.abc.Callable
from collections import defaultdict
import os
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings(action='ignore')

def fontRegistered():
    font_dirs = [os.getcwd() + '/customFonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

fontRegistered()

plt.rc('font', family='NanumGothic')

# KBO 팀 데이터 가져오기
def search_team(teamname, year):
    teamcode = {
        '해태': 2001, 'KIA': 2002, '삼성': 1001, 'OB': 6001, '두산': 6002,
        'SK': 9001, 'SSG': 9002, '삼미': 4001, '청보': 4002, '태평양': 4003,
        '현대': 4004, 'MBC': 5001, 'LG': 5002, '롯데': 3001, '빙그레': 7001,
        '한화': 7002, 'NC': 11001, 'KT': 12001, '우리': 10001, '서울': 10001,
        '넥센': 10001, '키움': 10001, '쌍방울': 8001
    }
    url = f'https://statiz.sporki.com/team/?m=seasonPosition&t_code={teamcode[teamname]}&year={year}'

    response = requests.get(url)

    if response.status_code == 200:
        pattern = re.compile('[ㄱ-ㅎ가-힣]+')
        positions = ['sp', 'rp', 'c', 'fb', 'sb', 'tb', 'ss', 'lf', 'cf', 'rf', 'dh']
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        team = {}

        for pos in positions:
            players = soup.select_one(f'body > div.warp > div.container > section > div.top_meum_box > div.box_type_boared.flex_box > div:nth-of-type(2) > div > div > div.box_cont > div > div.{pos}.player_m')
            players_list = pattern.findall(str(players))
            players_list_with_detail = []
            for player in players_list:
                players_list_with_detail.append([player, str(year), teamname])
            team[pos] = players_list_with_detail
        return team

    else:
        print('error')

# 가중치를 두어 팀별 평균 지표 계산 함수
def calculate_weighted_averages(data, is_pitcher=False):
    data = data.copy()  # 원본 데이터를 변경하지 않도록 복사
    if is_pitcher:
        data['Weight'] = data['IP'].astype(float)
    else:
        data['Weight'] = data['G'].astype(float)

    weighted_avg = {}
    for col in ['OPS', 'SLG', 'OBP', 'ERA', 'WHIP']:
        if col in data.columns:
            data[col] = data[col].replace('', 0).astype(float)  # 빈 문자열을 0으로 대체하고 float로 변환
            weighted_avg[col] = (data[col] * data['Weight']).sum() / data['Weight'].sum()

    return pd.Series(weighted_avg)

# 승리 확률 계산 함수
def calculate_win_probability(team1_stats, team2_stats, model, scaler):
    combined_stats = list(team1_stats) + list(team2_stats)
    combined_stats_scaled = scaler.transform([combined_stats])
    win_prob = model.predict_proba(combined_stats_scaled)[0][1]
    return win_prob

# 팀 간 경기 시뮬레이션
def simulate_season(team_stats_df, model, scaler, num_games=144, num_simulations=10):
    teams = team_stats_df.index
    total_results = {team: {'wins': 0, 'losses': 0} for team in teams}

    for _ in range(num_simulations):
        results = {team: {'wins': 0, 'losses': 0} for team in teams}
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i >= j:
                    continue
                team1_wins = 0
                team2_wins = 0
                for _ in range(num_games // (len(teams) - 1)):
                    win_prob = calculate_win_probability(team_stats_df.loc[team1], team_stats_df.loc[team2], model, scaler)
                    if random.random() < win_prob:
                        team1_wins += 1
                    else:
                        team2_wins += 1
                results[team1]['wins'] += team1_wins
                results[team1]['losses'] += team2_wins
                results[team2]['wins'] += team2_wins
                results[team2]['losses'] += team1_wins
        for team in teams:
            total_results[team]['wins'] += results[team]['wins']
            total_results[team]['losses'] += results[team]['losses']

    # 평균값 계산
    for team in teams:
        total_results[team]['wins'] /= num_simulations
        total_results[team]['losses'] /= num_simulations

    return total_results


# Streamlit 앱 시작
st.set_page_config(page_title='KBO리그 시뮬레이션')

# 페이지 설정
pages = ["홈", "데이터 수집 과정", "시뮬레이션 과정", "선수 성적 일람", "팀 편집", "예상 순위"]
page = st.sidebar.selectbox("페이지 선택", pages)

if page == "홈":
    st.markdown("""
    # KBO리그 시뮬레이션
    이 앱은 MLB 데이터와 KBO 데이터를 기반으로 팀 성적을 예측하는 시뮬레이션 앱입니다.

    ## 사용된 주요 야구 지표
    - **OPS (On-base Plus Slugging)**: 타자의 출루율과 장타율을 합한 값
        - 수식: OPS = 출루율 (OBP) + 장타율 (SLG)
    - **SLG (Slugging Percentage)**: 타자의 장타율
        - 수식: SLG = (안타 (1B) + 2루타 (2B) * 2 + 3루타 (3B) * 3 + 홈런 (HR) * 4) / 타수 (AB)
    - **OBP (On-base Percentage)**: 타자의 출루율
        - 수식: OBP = (안타 (H) + 볼넷 (BB) + 사구 (HBP)) / (타수 (AB) + 볼넷 (BB) + 사구 (HBP) + 희생플라이 (SF))
    - **ERA (Earned Run Average)**: 투수의 평균 자책점
        - 수식: ERA = (자책점 (ER) * 9) / 이닝 (IP)
    - **WHIP (Walks plus Hits per Inning Pitched)**: 이닝당 출루 허용률
        - 수식: WHIP = (볼넷 (BB) + 피안타 (H)) / 이닝 (IP)

    ## 기능 설명
    1. **선수 성적 일람**: CSV 파일에 있는 선수들의 성적을 일람할 수 있습니다.
    2. **팀 편집**: 각 팀의 라인업을 편집할 수 있습니다.
    3. **예상 순위**: 편집한 팀 라인업을 기반으로 KBO리그의 예상 순위를 예측합니다.

    ## 사용 방법
    1. 사이드바에서 원하는 페이지를 선택하세요.
    2. 선수 성적 일람 페이지에서 선수들의 성적을 확인하세요.
    3. 팀 편집 페이지에서 각 팀의 라인업을 편집하세요.
    4. 예상 순위 페이지에서 시뮬레이션 결과를 확인하세요.
    """)

elif page == "데이터 수집 과정":
    st.title("데이터 수집 과정")
    
    st.write("아래의 코드로 2013~2023년 KBO 선수들의 데이터를 수집해 csv파일로 저장합니다.")

    st.code('''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.alert import Alert
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import collections
collections.Callable = collections.abc.Callable
from bs4 import BeautifulSoup
from html_table_parser import parser_functions as parser
import pandas as pd
import re
import time
import queue
import os

team_name_changes = {
    2013: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC'],
    2014: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC'],
    2015: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC'],
    2016: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC'],
    2017: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC', 'KT'],
    2018: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '넥센', 'NC', 'KT'],
    2019: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '키움', 'NC', 'KT'],
    2020: ['LG', 'KIA', '삼성', '두산', '롯데', 'SK', '한화', '키움', 'NC', 'KT'],
    2021: ['LG', 'KIA', '삼성', '두산', '롯데', 'SSG', '한화', '키움', 'NC', 'KT'],
    2022: ['LG', 'KIA', '삼성', '두산', '롯데', 'SSG', '한화', '키움', 'NC', 'KT'],
    2023: ['LG', 'KIA', '삼성', '두산', '롯데', 'SSG', '한화', '키움', 'NC', 'KT']
}

# Driver pool
driver_pool = queue.Queue()

def initialize_driver_pool(size=10):
    for _ in range(size):
        driver = get_driver()
        driver_pool.put(driver)

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--ignore-ssl-errors=yes')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-gpu')
    options.add_argument('--incognito')
    options.add_argument('--disable-images')
    options.add_argument('--blink-settings=imagesEnabled=false')
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.200'
    options.add_argument(f'user-agent={user_agent}')
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(2)
    return driver

def get_driver_from_pool():
    return driver_pool.get()

def return_driver_to_pool(driver):
    driver_pool.put(driver)

def safe_click(driver, by, value, timeout=20, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            element = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            element.click()
            return
        except (ElementClickInterceptedException, StaleElementReferenceException, TimeoutException):
            attempt += 1
            time.sleep(0.5)
    raise TimeoutException(f"Failed to click element after {retries} attempts")

def handle_alert(driver):
    try:
        WebDriverWait(driver, 2).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        alert_text = alert.text
        print(f"Alert text: {alert_text}")
        alert.accept()
    except TimeoutException:
        pass

def search_player(driver, name, year, team, is_pitcher):
    try:
        print(name)
        driver.get('https://statiz.sporki.com/')
        handle_alert(driver)
        safe_click(driver, By.XPATH, '/html/body/div[2]/header/div[3]/div[1]/a')
        handle_alert(driver)
        driver.find_element(By.NAME, 's').send_keys(name)
        driver.find_element(By.XPATH, '/html/body/div[2]/div[3]/form/div/input[2]').click()
        handle_alert(driver)

        url = driver.current_url
        if 'search' in url:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            p = parser.make2d(temp[0])
            df = pd.DataFrame(p)
            l = len(df.index)
            for i in range(1, l):
                try:
                    player_link = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, f'/html/body/div[2]/div[6]/section/div[3]/div/table/tbody/tr[{i}]/td[1]/a'))
                    )
                    player_link.click()
                    handle_alert(driver)
                    teamhist = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[6]/section/div[3]/div[1]/ul/li[5]'))
                    ).text
                    pos = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '/html/body/div[2]/div[6]/section/div[3]/div[1]/div[3]/div[2]/span[2]'))
                    ).text
                    if team in teamhist and (pos == 'P' if is_pitcher else pos != 'P'):
                        break
                    else:
                        driver.back()
                except (TimeoutException, NoSuchElementException):
                    driver.back()

        safe_click(driver, By.XPATH, '/html/body/div[2]/div[6]/section/div[2]/div[2]/ul/li[2]/a')
        handle_alert(driver)
        try:    
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            p = parser.make2d(temp[0])
            df = pd.DataFrame(p[2:-3], columns=p[0])
            columns = list(df.columns)
            df = df[df.Year == year]

            # Clean data and convert to float
            df.replace('', pd.NA, inplace=True)
            df.dropna(inplace=True)
            df = df.astype('float', errors='ignore')

            if is_pitcher:
                df = df[['IP', 'ERA', 'WHIP']]
            else:
                columns[26:32] = ['AVG', 'OBP', 'SLG', 'OPS', 'R/ePA', 'wRC+']
                df.columns = columns
                df = df[['G', 'OPS', 'SLG', 'OBP']]

            # 선수 이름과 연도 추가
            df['Player'] = name
            df['Year'] = year
            df['Team'] = team

            print(f'{name} complete')
            return df
        except:
            print(f'{name} passed')
            return pd.DataFrame()
    except WebDriverException as e:
        print(f"Skipping {name} due to error: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

def search_team(teamname, year):
    teamcode = {
        '해태': 2001, 'KIA': 2002, '삼성': 1001, 'OB': 6001, '두산': 6002,
        'SK': 9001, 'SSG': 9002, '삼미': 4001, '청보': 4002, '태평양': 4003,
        '현대': 4004, 'MBC': 5001, 'LG': 5002, '롯데': 3001, '빙그레': 7001,
        '한화': 7002, 'NC': 11001, 'KT': 12001, '우리': 10001, '서울': 10001,
        '넥센': 10001, '키움': 10001, '쌍방울': 8001
    }
    url = f'https://statiz.sporki.com/team/?m=seasonPosition&t_code={teamcode[teamname]}&year={year}'

    response = requests.get(url)

    if response.status_code == 200:
        pattern = re.compile('[ㄱ-ㅎ가-힣]+')
        positions = ['sp', 'rp', 'c', 'fb', 'sb', 'tb', 'ss', 'lf', 'cf', 'rf', 'dh']
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        team = {}

        for pos in positions:
            players = soup.select_one(f'body > div.warp > div.container > section > div.top_meum_box > div.box_type_boared.flex_box > div:nth-of-type(2) > div > div > div.box_cont > div > div.{pos}.player_m')
            players_list = pattern.findall(str(players))
            players_list_with_detail = []
            for player in players_list:
                players_list_with_detail.append([player, str(year), teamname])
            team[pos] = players_list_with_detail
        return team

    else:
        print('error')

def fetch_data(player, year, team, is_pitcher):
    driver = get_driver_from_pool()
    try:
        data = search_player(driver, player, year, team, is_pitcher)
    finally:
        return_driver_to_pool(driver)
    return data

def process_players(players, is_pitcher_flags):
    data = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_data, player[0], player[1], player[2], is_pitcher) for player, is_pitcher in zip(players, is_pitcher_flags)]
        for future in as_completed(futures):
            player_data = future.result()
            data = pd.concat([data, player_data], ignore_index=True)
    return data

def fetch_team_data(teamname, year):
    teamlist = search_team(teamname, year)
    players = teamlist['sp'] + teamlist['rp'] + teamlist['c'] + teamlist['fb'] + teamlist['sb'] + teamlist['tb'] + teamlist['ss'] + teamlist['lf'] + teamlist['cf'] + teamlist['rf'] + teamlist['dh']
    is_pitcher_flags = [True] * (len(teamlist['sp']) + len(teamlist['rp'])) + [False] * (len(players) - (len(teamlist['sp']) + len(teamlist['rp'])))
    players_data = process_players(players, is_pitcher_flags)
    return players_data

def teams_average_data(teams, year):
    initialize_driver_pool(10)

    team_data_list = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_team_data, teamname, year) for teamname in teams]
        for future in as_completed(futures):
            team_data = future.result()
            team_data_list.append(team_data)

    all_teams_data = pd.concat(team_data_list, ignore_index=True)

    # Cleanup drivers
    while not driver_pool.empty():
        driver = driver_pool.get()
        driver.quit()

    return all_teams_data

driver_pool = queue.Queue()

for year in range(2013, 2024):
    teams = team_name_changes[year]
    data = teams_average_data(teams, year)
    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv(f'data/player_data_{year}.csv', index=False)
    print(f'Saved player data for {year}')

print("Data collection complete.")
''', language='python')

    st.write("""
    아래 버튼을 클릭하여 2013~2023년까지의 데이터를 하나로 합쳐놓은 데이터셋을 다운로드하세요.
    """)
    
    # 데이터셋 다운로드 버튼
    file_path = 'data/player_data_2013~2023.csv'
    with open(file_path, 'rb') as f:
        st.download_button(label='player_data_2013~2023.csv 다운로드', data=f, file_name='player_data_2013~2023.csv', mime='text/csv')


# 시뮬레이션 과정 페이지
elif page == "시뮬레이션 과정":
    st.title("시뮬레이션 과정")
    st.write("""
    이 앱은 KBO 리그의 팀 성적을 예측하기 위해 머신러닝 모델을 사용합니다. 
    다음은 모델을 학습시키고 리그 시뮬레이션을 수행하는 과정입니다.
    """)

    st.write("### 1. 데이터 수집 및 전처리")
    st.write("먼저, MLB API를 사용하여 팀 정보와 일정 데이터를 수집합니다.")
    st.code("""
import requests
import pandas as pd
from datetime import datetime, timedelta

# MLB Stats API 기본 URL
BASE_URL = "https://statsapi.mlb.com/api/v1/"

# 팀 정보를 가져오는 함수
def get_teams():
    url = BASE_URL + "teams?sportId=1"  # 전체 MLB 팀 정보를 가져오는 URL
    response = requests.get(url)  # API 요청
    teams = response.json()['teams']  # 응답에서 팀 정보 추출
    team_dict = {team['id']: team['name'] for team in teams}  # 팀 ID와 이름을 딕셔너리로 변환
    return team_dict  # 팀 딕셔너리 반환

# 특정 기간 동안의 경기 일정을 가져오는 함수
def get_schedule(start_date, end_date):
    url = BASE_URL + f"schedule?startDate={start_date}&endDate={end_date}&sportId=1"  # 일정 정보를 가져오는 URL
    response = requests.get(url)  # API 요청
    games = response.json()['dates']  # 응답에서 일정 정보 추출
    game_list = []
    for date in games:
        for game in date['games']:
            game_list.append({
                'gamePk': game['gamePk'],  # 게임 ID
                'gameDate': game['gameDate'],  # 경기 날짜
                'homeTeam': game['teams']['home']['team']['name'],  # 홈 팀 이름
                'awayTeam': game['teams']['away']['team']['name']  # 원정 팀 이름
            })
    return game_list  # 경기 리스트 반환
    """, language="python")

    st.write("이제 각 경기의 데이터를 수집하고, 이를 전처리하여 주요 지표를 추출합니다.")
    st.code("""
# 특정 게임의 상세 데이터를 가져오는 함수
def get_game_data(gamePk):
    url = BASE_URL + f"game/{gamePk}/boxscore"  # 특정 게임의 박스스코어 정보를 가져오는 URL
    response = requests.get(url)  # API 요청
    game_data = response.json()  # 응답에서 게임 데이터 추출
    return game_data  # 게임 데이터 반환

# 게임 데이터를 가공하는 함수
def process_game_data(game_data):
    gamePk = game_data.get('gamePk', 'N/A')  # 게임 ID
    gameDate = game_data.get('gameDate', 'N/A')  # 경기 날짜

    teams = game_data.get('teams', {})
    home_team_data = teams.get('home', {})  # 홈 팀 데이터
    away_team_data = teams.get('away', {})  # 원정 팀 데이터

    home_team = home_team_data.get('team', {}).get('name', 'N/A')  # 홈 팀 이름
    away_team = away_team_data.get('team', {}).get('name', 'N/A')  # 원정 팀 이름

    home_stats = home_team_data.get('teamStats', {}).get('batting', {})  # 홈 팀 타격 데이터
    away_stats = away_team_data.get('teamStats', {}).get('batting', {})  # 원정 팀 타격 데이터

    home_ops = home_stats.get('ops', 0)  # 홈 팀 OPS
    home_slg = home_stats.get('slg', 0)  # 홈 팀 SLG
    home_obp = home_stats.get('obp', 0)  # 홈 팀 OBP

    home_pitching = home_team_data.get('teamStats', {}).get('pitching', {})  # 홈 팀 투구 데이터
    home_era = home_pitching.get('era', 0)  # 홈 팀 ERA
    home_whip = home_pitching.get('whip', 0)  # 홈 팀 WHIP

    away_ops = away_stats.get('ops', 0)  # 원정 팀 OPS
    away_slg = away_stats.get('slg', 0)  # 원정 팀 SLG
    away_obp = away_stats.get('obp', 0)  # 원정 팀 OBP

    away_pitching = away_team_data.get('teamStats', {}).get('pitching', {})  # 원정 팀 투구 데이터
    away_era = away_pitching.get('era', 0)  # 원정 팀 ERA
    away_whip = away_pitching.get('whip', 0)  # 원정 팀 WHIP

    home_runs = home_team_data.get('teamStats', {}).get('batting', {}).get('runs', 0)  # 홈 팀 득점
    away_runs = away_team_data.get('teamStats', {}).get('batting', {}).get('runs', 0)  # 원정 팀 득점
    home_wins = 1 if home_runs > away_runs else 0  # 홈 팀 승리 여부 (승리: 1, 패배: 0)

    return {
        'gamePk': gamePk,
        'gameDate': gameDate,
        'homeTeam': home_team,
        'awayTeam': away_team,
        'home_ops': home_ops,
        'home_slg': home_slg,
        'home_obp': home_obp,
        'home_era': home_era,
        'home_whip': home_whip,
        'away_ops': away_ops,
        'away_slg': away_slg,
        'away_obp': away_obp,
        'away_era': away_era,
        'away_whip': away_whip,
        'home_wins': home_wins
    }

# 시작 날짜와 종료 날짜 설정
start_date = "2023-04-01"
end_date = "2023-10-01"

# 경기 일정 가져오기
schedule = get_schedule(start_date, end_date)

# 팀 정보 가져오기
team_dict = get_teams()

# 샘플 게임 데이터 가져오기 (100개의 경기 데이터)
sample_games = schedule[:100]
game_data_list = []

# 각 게임의 상세 데이터 가져오기
for game in sample_games:
    game_data = get_game_data(game['gamePk'])
    game_data_list.append(game_data)

# 게임 데이터 가공
processed_game_data_list = [process_game_data(game_data) for game_data in game_data_list]

# 가공된 데이터를 데이터프레임으로 변환
df_processed_games = pd.DataFrame(processed_game_data_list)
    """, language="python")

    st.write("### 2. 모델 학습")
    st.write("수집한 데이터를 바탕으로 랜덤 포레스트 모델을 학습시킵니다.")
    st.code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# 특징 및 타겟 변수 생성
X = df_processed_games[['home_ops', 'home_slg', 'home_obp', 'home_era', 'home_whip',
                        'away_ops', 'away_slg', 'away_obp', 'away_era', 'away_whip']]  # 입력 변수
y = df_processed_games['home_wins']  # 타겟 변수 (홈 팀의 승리 여부)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 데이터셋을 학습용과 테스트용으로 분할

# 데이터 스케일링
scaler = StandardScaler()  # 표준화 스케일러 객체 생성
X_train_scaled = scaler.fit_transform(X_train)  # 학습용 데이터를 스케일링하고 변환
X_test_scaled = scaler.transform(X_test)  # 테스트용 데이터를 스케일링하고 변환

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)  # 랜덤 포레스트 분류기 객체 생성
model.fit(X_train_scaled, y_train)  # 학습용 데이터를 사용하여 모델 학습

# 예측 및 평가
y_pred = model.predict(X_test_scaled)  # 테스트용 데이터로 예측 수행
accuracy = accuracy_score(y_test, y_pred)  # 예측 결과와 실제 값을 비교하여 정확도 계산
print(f'Accuracy: {accuracy}')  # 정확도 출력

# 모델과 스케일러 저장
with open('rf_model.pkl', 'wb') as f:  # 랜덤 포레스트 모델을 파일에 저장
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:  # 스케일러 객체를 파일에 저장
    pickle.dump(scaler, f)
    """, language="python")

    st.write("### 랜덤 포레스트 모델이란?")
    st.write("""
    랜덤 포레스트(Random Forest)는 앙상블 학습 방법 중 하나로, 다수의 결정 트리(decision tree)를 학습하고 예측을 합치는 방식으로 동작합니다.
    
    주요 특징:
    - 여러 개의 결정 트리를 학습하여 예측 성능을 향상시킵니다.
    - 개별 트리의 예측을 종합하여 최종 예측을 도출합니다.
    - 과적합(overfitting)을 방지하는 데 효과적입니다.
    
    랜덤 포레스트는 각 트리가 학습할 때 사용하는 데이터 샘플을 무작위로 선택하고, 트리의 노드에서 분할할 특징을 무작위로 선택합니다. 
    이를 통해 다수의 약한 학습기를 결합하여 강력한 학습기를 만드는 원리를 사용합니다.
    """)

    st.write("### 3. 리그 시뮬레이션")
    st.write("학습된 모델을 사용하여 리그 시뮬레이션을 실행합니다.")
    st.code("""
# 승리 확률 계산 함수
def calculate_win_probability(team1_stats, team2_stats, model, scaler):
    # 두 팀의 통계 데이터를 결합하여 하나의 리스트로 만듦
    combined_stats = list(team1_stats) + list(team2_stats)
    # 결합된 통계 데이터를 스케일러를 사용하여 스케일링
    combined_stats_scaled = scaler.transform([combined_stats])
    # 스케일링된 데이터를 모델에 입력하여 승리 확률 예측 (0과 1 사이의 확률 값 반환)
    win_prob = model.predict_proba(combined_stats_scaled)[0][1]
    return win_prob  # 팀1의 승리 확률 반환

# 팀 간 경기 시뮬레이션
def simulate_season(team_stats_df, model, scaler, num_games=144, num_simulations=50):
    teams = team_stats_df.index  # 팀 목록
    # 각 팀의 승리 및 패배 수를 저장할 딕셔너리 초기화
    total_results = {team: {'wins': 0, 'losses': 0} for team in teams}

    # 지정된 시뮬레이션 횟수만큼 반복
    for _ in range(num_simulations):
        # 각 시뮬레이션에서의 결과를 저장할 딕셔너리 초기화
        results = {team: {'wins': 0, 'losses': 0} for team in teams}
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i >= j:
                    continue  # 같은 팀 간의 경기는 건너뜀
                team1_wins = 0
                team2_wins = 0
                # 각 팀 간의 경기를 지정된 횟수만큼 반복
                for _ in range(num_games // (len(teams) - 1)):
                    # 팀1과 팀2 간의 승리 확률 계산
                    win_prob = calculate_win_probability(team_stats_df.loc[team1], team_stats_df.loc[team2], model, scaler)
                    # 무작위 수를 생성하여 승리 팀 결정
                    if random.random() < win_prob:
                        team1_wins += 1
                    else:
                        team2_wins += 1
                # 경기 결과를 각 팀의 승리 및 패배 수에 추가
                results[team1]['wins'] += team1_wins
                results[team1]['losses'] += team2_wins
                results[team2]['wins'] += team2_wins
                results[team2]['losses'] += team1_wins
        # 각 시뮬레이션의 결과를 총 결과에 합산
        for team in teams:
            total_results[team]['wins'] += results[team]['wins']
            total_results[team]['losses'] += results[team]['losses']

    # 각 팀의 평균 승리 및 패배 수 계산
    for team in teams:
        total_results[team]['wins'] /= num_simulations
        total_results[team]['losses'] /= num_simulations

    return total_results  # 최종 결과 반환

# 가중치를 두어 팀별 평균 지표 계산 함수
def calculate_weighted_averages(data, is_pitcher=False):
    data = data.copy()  # 원본 데이터를 변경하지 않도록 복사
    if is_pitcher:
        data['Weight'] = data['IP'].astype(float)  # 투수일 경우 소화 이닝(IP)을 가중치로 설정
    else:
        data['Weight'] = data['G'].astype(float)  # 타자일 경우 경기 수(G)를 가중치로 설정

    weighted_avg = {}
    for col in ['OPS', 'SLG', 'OBP', 'ERA', 'WHIP']:  # 계산할 지표 목록
        if col in data.columns:
            data[col] = data[col].replace('', 0).astype(float)  # 빈 문자열을 0으로 대체하고 float로 변환
            # 가중 평균 계산: 각 지표 값에 가중치를 곱한 후 합산하고, 가중치의 총합으로 나눔
            weighted_avg[col] = (data[col] * data['Weight']).sum() / data['Weight'].sum()

    return pd.Series(weighted_avg)  # 가중 평균 값을 시리즈로 반환

# 각 팀의 선수 명단 가져오기
teams = ['LG', 'KT', 'SSG', 'NC', '두산', 'KIA', '롯데', '한화', '삼성', '키움']
team_data_dict = {}

for team in teams:
    team_data_dict[team] = search_team(team, 2023)  # 각 팀의 선수 명단을 딕셔너리에 저장

# KBO 데이터 로드
file_path = '/content/player_data_2023.csv'  # CSV 파일 경로
kbo_player_data = pd.read_csv(file_path)

# 각 팀의 선수 데이터를 이용해 평균 지표 계산
team_stats = {}

for team, players in team_data_dict.items():
    team_players = []
    for position, player_list in players.items():
        for player in player_list:
            player_name, player_year, player_team = player
            # 선수의 이름과 연도로 데이터를 필터링하여 해당 선수의 기록을 가져옴
            player_stats = kbo_player_data[(kbo_player_data['Player'] == player_name) & (kbo_player_data['Year'] == int(player_year))]
            if not player_stats.empty:
                team_players.append(player_stats)
    if team_players:
        team_data = pd.concat(team_players, ignore_index=True)  # 각 선수의 데이터를 하나의 데이터프레임으로 병합
        hitters = team_data[team_data['OPS'].notna()]  # 타자 데이터 필터링 (OPS 값이 있는 데이터)
        pitchers = team_data[team_data['ERA'].notna()]  # 투수 데이터 필터링 (ERA 값이 있는 데이터)

        if not hitters.empty:
            hitters_avg = calculate_weighted_averages(hitters, is_pitcher=False)  # 타자 평균 지표 계산
        else:
            hitters_avg = pd.Series({'OPS': 0, 'SLG': 0, 'OBP': 0})  # 타자 데이터가 없는 경우 0으로 채움

        if not pitchers.empty:
            pitchers_avg = calculate_weighted_averages(pitchers, is_pitcher=True)  # 투수 평균 지표 계산
        else:
            pitchers_avg = pd.Series({'ERA': 0, 'WHIP': 0})  # 투수 데이터가 없는 경우 0으로 채움

        team_avg = pd.concat([hitters_avg, pitchers_avg])  # 타자와 투수의 평균 지표 병합
        team_stats[team] = team_avg  # 팀별 평균 지표를 딕셔너리에 저장

team_stats_df = pd.DataFrame(team_stats).T  # 딕셔너리를 데이터프레임으로 변환
team_stats_df = team_stats_df.loc[:, (team_stats_df != 0).any(axis=0)]  # 불필요한 0으로 된 칼럼 삭제

# 모델과 스케일러 로드
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 시즌 시뮬레이션 실행
results = simulate_season(team_stats_df, model, scaler)

# 결과 출력 및 정렬
results_df = pd.DataFrame(results).T  # 결과를 데이터프레임으로 변환
results_df = results_df.sort_values(by='wins', ascending=False)  # 승리 수를 기준으로 정렬
results_df['rank'] = range(1, len(results_df) + 1)  # 순위를 계산하여 추가

# 순위표 출력
print(results_df)

# 시각화
plt.figure(figsize=(12, 8))
plt.bar(results_df.index, results_df['wins'], color='b', alpha=0.7, label='Wins')
plt.bar(results_df.index, results_df['losses'], bottom=results_df['wins'], color='r', alpha=0.7, label='Losses')
plt.xlabel('Teams')
plt.ylabel('Number of Games')
plt.title('Simulated KBO Season Results')
plt.legend()
plt.xticks(rotation=45)
plt.show()
    """, language="python")


elif page == "선수 성적 일람":
    st.title("선수 성적 일람")
    
    # 제공된 CSV 파일 경로
    file_path = 'data/player_data_2013~2023.csv'
    player_data = pd.read_csv(file_path)
    
    # Player 값이 "Player"인 행 삭제
    player_data = player_data[player_data['Player'] != 'Player']
    
    # 지표들을 숫자로 변환
    for col in ['OPS', 'SLG', 'OBP', 'ERA', 'WHIP', 'G', 'IP']:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce')
    
    # 타자와 투수 분리
    hitters = player_data.dropna(subset=['OPS', 'SLG', 'OBP'])
    pitchers = player_data.dropna(subset=['ERA', 'WHIP'])

    # 타자 지표 표시
    st.header("타자 성적 일람")
    st.dataframe(hitters[['Player', 'Year', 'Team', 'G', 'OPS', 'SLG', 'OBP']], width=1200)

    # 투수 지표 표시
    st.header("투수 성적 일람")
    st.dataframe(pitchers[['Player', 'Year', 'Team', 'IP', 'ERA', 'WHIP']], width=1200)

    # 각 팀별 성적 요약
    teams = player_data['Team'].unique()[:-1]
    st.header("팀별 선수 일람")
    selected_team = st.selectbox("팀 선택", teams)
    if selected_team:
        team_hitters = hitters[hitters['Team'] == selected_team]
        team_pitchers = pitchers[pitchers['Team'] == selected_team]
        st.subheader(f"{selected_team} 타자 성적")
        st.dataframe(team_hitters[['Player', 'Year', 'Team', 'G', 'OPS', 'SLG', 'OBP']], width=1200)
        st.subheader(f"{selected_team} 투수 성적")
        st.dataframe(team_pitchers[['Player', 'Year', 'Team', 'IP', 'ERA', 'WHIP']], width=1200)


# 팀 라인업 편집 페이지
elif page == "팀 편집":
    st.title("팀 라인업 편집")
    
    # 제공된 CSV 파일 경로
    file_path = 'data/player_data_2013~2023.csv'
    player_data = pd.read_csv(file_path)

    # Player 값이 "Player"인 행 삭제
    player_data = player_data[player_data['Player'] != 'Player']
    
    # 팀 목록
    teams = ['LG', 'KT', 'SSG', 'NC', '두산', 'KIA', '롯데', '한화', '삼성', '키움']
    selected_team = st.selectbox("팀 선택", teams)
    
    # 모든 팀의 선수 정보
    all_team_players = player_data.copy()
    
    # 기본 라인업 설정
    @st.cache_data
    def get_default_lineup(team, year=2023):
        team_lineup = search_team(team, year)
        return team_lineup
    
    # 세션 상태 초기화
    if "team_lineup" not in st.session_state:
        st.session_state.team_lineup = {team: get_default_lineup(team) for team in teams}
    
    if "lineup_state" not in st.session_state:
        st.session_state.lineup_state = defaultdict(lambda: defaultdict(list))
        for team in teams:
            default_lineup = get_default_lineup(team)
            for pos, players in default_lineup.items():
                st.session_state.lineup_state[team][pos] = [[player[0], player[1]] for player in players]
    
    team_lineup = st.session_state.lineup_state[selected_team]
    
    positions = ['sp', 'rp', 'c', 'fb', 'sb', 'tb', 'ss', 'lf', 'cf', 'rf', 'dh']
    pos_name = {
        'sp' : '선발 투수',
        'rp' : '구원 투수',
        'c' : '포수',
        'fb' : '1루수',
        'sb' : '2루수',
        'tb' : '3루수',
        'ss' : '유격수',
        'lf' : '좌익수',
        'cf' : '중견수',
        'rf' : '우익수',
        'dh' : '지명타자'
    }
    
    # 포지션별 선수 입력 필드
    for pos in positions:
        st.write(f"### {pos_name[pos]}")
        available_players = player_data.dropna(subset=['OPS', 'SLG', 'OBP']) if pos not in ['sp', 'rp'] else player_data.dropna(subset=['ERA', 'WHIP'])
        for idx, player in enumerate(team_lineup[pos]):
            current_player = player[0] if player[0] in available_players['Player'].values else '선수 선택'
            years = sorted(player_data[player_data['Player'] == current_player]['Year'].unique(), reverse=True) if current_player != '선수 선택' else []
            current_year = player[1] if player[1] in years else (years[0] if years else '연도 선택')
            
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                selected_player = st.selectbox(f"{pos_name[pos]} {idx+1}", ['선수 선택'] + list(available_players['Player'].values), index=0 if current_player == '선수 선택' else available_players['Player'].values.tolist().index(current_player) + 1, key=f"{pos}_{idx}_player")
                if selected_player != team_lineup[pos][idx][0]:
                    team_lineup[pos][idx][0] = selected_player
                    # 선수 변경 시 연도 자동 업데이트
                    years = sorted(player_data[player_data['Player'] == selected_player]['Year'].unique(), reverse=True) if selected_player != '선수 선택' else []
                    if years:
                        team_lineup[pos][idx][1] = years[0]
                    st.experimental_rerun()
            with col2:
                if selected_player != '선수 선택':
                    try:
                        selected_year = st.selectbox(f"연도 {idx+1}", years, index=0 if current_year == '연도 선택' else years.index(current_year), key=f"{pos}_{idx}_year")
                    except ValueError:
                        selected_year = years[0]
                        team_lineup[pos][idx][1] = selected_year
                    team_lineup[pos][idx][1] = selected_year
            with col3:
                if len(team_lineup[pos]) > 1 or pos == 'dh':
                    if st.button('삭제', key=f"remove_{pos}_{idx}"):
                        team_lineup[pos].pop(idx)
                        st.experimental_rerun()
        
        # 새로운 선수 추가
        if st.button(f"선수 추가", key=f"add_{pos}"):
            default_player = '구자욱' if pos not in ['sp', 'rp'] else '김광현'
            default_year = '2023'
            team_lineup[pos].append([default_player, default_year])
            st.experimental_rerun()

    # 선수 이름이 "선수 선택"인 칸을 모두 삭제
    for pos in positions:
        team_lineup[pos] = [player for player in team_lineup[pos] if player[0] != '선수 선택']
    
    # 변경 사항 저장
    st.session_state.lineup_state[selected_team] = team_lineup



elif page == "예상 순위":
    st.title("예상 순위표")

    # 로딩 화면 추가
    with st.spinner('팀 데이터를 불러오고 있습니다...'):
        # 각 팀의 선수 명단 가져오기
        teams = ['LG', 'KT', 'SSG', 'NC', '두산', 'KIA', '롯데', '한화', '삼성', '키움']
        
        # 기본 라인업 설정
        @st.cache_data
        def get_default_lineup(team, year=2023):
            team_lineup = search_team(team, year)
            return team_lineup
        
        # 세션 상태 초기화
        if "team_lineup" not in st.session_state:
            st.session_state.team_lineup = {team: get_default_lineup(team) for team in teams}
        
        if "lineup_state" not in st.session_state:
            st.session_state.lineup_state = defaultdict(lambda: defaultdict(list))
            for team in teams:
                default_lineup = get_default_lineup(team)
                for pos, players in default_lineup.items():
                    st.session_state.lineup_state[team][pos] = [[player[0], player[1]] for player in players]
    
    # KBO 데이터 로드
    file_path = 'data/player_data_2013~2023.csv'
    kbo_player_data = pd.read_csv(file_path)

    # 각 팀의 선수 데이터를 이용해 평균 지표 계산
    team_stats = {}

    for team in teams:
        team_players = []
        for pos, player_list in st.session_state.lineup_state[team].items():
            for player in player_list:
                player_name, player_year = player
                player_stats = kbo_player_data[(kbo_player_data['Player'] == player_name) & (kbo_player_data['Year'] == player_year)]
                if not player_stats.empty:
                    team_players.append(player_stats)
        if team_players:
            team_data = pd.concat(team_players, ignore_index=True)
            hitters = team_data[team_data['OPS'].notna()]
            pitchers = team_data[team_data['ERA'].notna()]

            if not hitters.empty:
                hitters_avg = calculate_weighted_averages(hitters, is_pitcher=False)
            else:
                hitters_avg = pd.Series({'OPS': 0, 'SLG': 0, 'OBP': 0})

            if not pitchers.empty:
                pitchers_avg = calculate_weighted_averages(pitchers, is_pitcher=True)
            else:
                pitchers_avg = pd.Series({'ERA': 0, 'WHIP': 0})

            team_avg = pd.concat([hitters_avg, pitchers_avg])
            team_stats[team] = team_avg

    team_stats_df = pd.DataFrame(team_stats).T
    team_stats_df = team_stats_df.loc[:, (team_stats_df != 0).any(axis=0)]  # 불필요한 0으로 된 칼럼 삭제

    # 모델과 스케일러 로드
    with st.spinner('모델과 스케일러를 로드하고 있습니다...'):
        with open('data/rf_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

    # 시즌 시뮬레이션 실행
    with st.spinner('시즌 시뮬레이션을 실행하고 있습니다...'):
        results = simulate_season(team_stats_df, model, scaler)

    # 결과 출력 및 정렬
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by='wins', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)

    # 순위표 출력
    st.write("## 시뮬레이션 결과")
    st.write(results_df)

    # 팀 평균 지표 출력
    st.write("## 팀 평균 지표")
    st.dataframe(team_stats_df)

    # 승리와 패배를 막대 그래프로 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(results_df.index, results_df['wins'], color='b', alpha=0.7, label='Wins')
    ax.bar(results_df.index, results_df['losses'], bottom=results_df['wins'], color='r', alpha=0.7, label='Losses')
    ax.set_xlabel('Teams')
    ax.set_ylabel('Number of Games')
    ax.set_title('Simulated KBO Season Results')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 각 팀별 OPS, SLG, OBP 평균을 히스토그램으로 시각화
    st.write("### 팀별 OPS, SLG, OBP 평균")
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    team_stats_df['OPS'].plot(kind='bar', ax=axs[0], color='blue', alpha=0.7)
    axs[0].set_title('Team OPS')
    axs[0].set_ylabel('OPS')

    team_stats_df['SLG'].plot(kind='bar', ax=axs[1], color='green', alpha=0.7)
    axs[1].set_title('Team SLG')
    axs[1].set_ylabel('SLG')

    team_stats_df['OBP'].plot(kind='bar', ax=axs[2], color='purple', alpha=0.7)
    axs[2].set_title('Team OBP')
    axs[2].set_ylabel('OBP')

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 각 팀별 ERA, WHIP 평균을 히스토그램으로 시각화
    st.write("### 팀별 ERA, WHIP 평균")
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    team_stats_df['ERA'].plot(kind='bar', ax=axs[0], color='red', alpha=0.7)
    axs[0].set_title('Team ERA')
    axs[0].set_ylabel('ERA')

    team_stats_df['WHIP'].plot(kind='bar', ax=axs[1], color='orange', alpha=0.7)
    axs[1].set_title('Team WHIP')
    axs[1].set_ylabel('WHIP')

    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 팀별 지표 박스 플롯
    st.write("### 팀별 지표 박스 플롯")
    fig, ax = plt.subplots(figsize=(12, 8))
    team_stats_df.boxplot(column=['OPS', 'SLG', 'OBP', 'ERA', 'WHIP'], ax=ax)
    ax.set_title('Team Stats Distribution')
    st.pyplot(fig)

