import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pickle
import matplotlib.pyplot as plt
import random
import collections
collections.Callable = collections.abc.Callable
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore')

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

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
pages = ["홈", "시뮬레이션 과정", "선수 성적 일람", "팀 편집", "예상 순위"]
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

# API 기본 URL
BASE_URL = "https://statsapi.mlb.com/api/v1/"

def get_teams():
    url = BASE_URL + "teams?sportId=1"
    response = requests.get(url)
    teams = response.json()['teams']
    team_dict = {team['id']: team['name'] for team in teams}
    return team_dict

def get_schedule(start_date, end_date):
    url = BASE_URL + f"schedule?startDate={start_date}&endDate={end_date}&sportId=1"
    response = requests.get(url)
    games = response.json()['dates']
    game_list = []
    for date in games:
        for game in date['games']:
            game_list.append({
                'gamePk': game['gamePk'],
                'gameDate': game['gameDate'],
                'homeTeam': game['teams']['home']['team']['name'],
                'awayTeam': game['teams']['away']['team']['name']
            })
    return game_list
    """, language="python")

    st.write("이제 각 경기의 데이터를 수집하고, 이를 전처리하여 주요 지표를 추출합니다.")
    st.code("""
def get_game_data(gamePk):
    url = BASE_URL + f"game/{gamePk}/boxscore"
    response = requests.get(url)
    game_data = response.json()
    return game_data

def process_game_data(game_data):
    gamePk = game_data.get('gamePk', 'N/A')
    gameDate = game_data.get('gameDate', 'N/A')

    teams = game_data.get('teams', {})
    home_team_data = teams.get('home', {})
    away_team_data = teams.get('away', {})

    home_team = home_team_data.get('team', {}).get('name', 'N/A')
    away_team = away_team_data.get('team', {}).get('name', 'N/A')

    home_stats = home_team_data.get('teamStats', {}).get('batting', {})
    away_stats = away_team_data.get('teamStats', {}).get('batting', {})

    home_ops = home_stats.get('ops', 0)
    home_slg = home_stats.get('slg', 0)
    home_obp = home_stats.get('obp', 0)

    home_pitching = home_team_data.get('teamStats', {}).get('pitching', {})
    home_era = home_pitching.get('era', 0)
    home_whip = home_pitching.get('whip', 0)

    away_ops = away_stats.get('ops', 0)
    away_slg = away_stats.get('slg', 0)
    away_obp = away_stats.get('obp', 0)

    away_pitching = away_team_data.get('teamStats', {}).get('pitching', {})
    away_era = away_pitching.get('era', 0)
    away_whip = away_pitching.get('whip', 0)

    home_runs = home_team_data.get('teamStats', {}).get('batting', {}).get('runs', 0)
    away_runs = away_team_data.get('teamStats', {}).get('batting', {}).get('runs', 0)
    home_wins = 1 if home_runs > away_runs else 0

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

start_date = "2023-04-01"
end_date = "2023-10-01"
schedule = get_schedule(start_date, end_date)
team_dict = get_teams()
sample_games = schedule[:5]
game_data_list = []

for game in sample_games:
    game_data = get_game_data(game['gamePk'])
    game_data_list.append(game_data)

processed_game_data_list = [process_game_data(game_data) for game_data in game_data_list]
df_processed_games = pd.DataFrame(processed_game_data_list)
print(df_processed_games.head())
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
                        'away_ops', 'away_slg', 'away_obp', 'away_era', 'away_whip']]
y = df_processed_games['home_wins']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 모델과 스케일러 저장
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
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
def calculate_win_probability(team1_stats, team2_stats, model, scaler):
    combined_stats = list(team1_stats) + list(team2_stats)
    combined_stats_scaled = scaler.transform([combined_stats])
    win_prob = model.predict_proba(combined_stats_scaled)[0][1]
    return win_prob

def simulate_season(team_stats_df, model, scaler, num_games=144, num_simulations=50):
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
                results[team1]['                losses'] += team2_wins
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

# 팀 평균 지표 계산 함수
def calculate_weighted_averages(data, is_pitcher=False):
    data = data.copy()
    if is_pitcher:
        data['Weight'] = data['IP'].astype(float)
    else:
        data['Weight'] = data['G'].astype(float)

    weighted_avg = {}
    for col in ['OPS', 'SLG', 'OBP', 'ERA', 'WHIP']:
        if col in data.columns:
            data[col] = data[col].replace('', 0).astype(float)
            weighted_avg[col] = (data[col] * data['Weight']).sum() / data['Weight'].sum()

    return pd.Series(weighted_avg)

# KBO 데이터 로드
file_path = 'data/player_data_2013~2023.csv'
kbo_player_data = pd.read_csv(file_path)

# 각 팀의 선수 데이터를 이용해 평균 지표 계산
teams = ['LG', 'KT', 'SSG', 'NC', '두산', 'KIA', '롯데', '한화', '삼성', '키움']
team_stats = {}

for team in teams:
    team_players = []
    for pos, player_list in st.session_state.lineup_state[team].items():
        for player in player_list:
            player_name, player_year = player
            player_stats = kbo_player_data[(kbo_player_data['Player'] == player_name) & (kbo_player_data['Year'] == int(player_year))]
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
team_stats_df = team_stats_df.loc[:, (team_stats_df != 0).any(axis=0)]

# 모델과 스케일러 로드
with open('data/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 시즌 시뮬레이션 실행
results = simulate_season(team_stats_df, model, scaler)

# 결과 출력 및 정렬
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='wins', ascending=False)
results_df['rank'] = range(1, len(results_df) + 1)

# 순위표 출력
st.write(results_df)

# 팀 평균 지표 출력
st.write("팀 평균 지표")
st.dataframe(team_stats_df)

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(results_df.index, results_df['wins'], color='b', alpha=0.7, label='Wins')
ax.bar(results_df.index, results_df['losses'], bottom=results_df['wins'], color='r', alpha=0.7, label='Losses')
ax.set_xlabel('Teams')
ax.set_ylabel('Number of Games')
ax.set_title('Simulated KBO Season Results')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
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
            team_lineup[pos].append(['선수 선택', '연도 선택'])
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
