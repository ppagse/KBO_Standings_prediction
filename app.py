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
pages = ["홈", "선수 성적 일람", "팀 편집", "예상 순위"]
page = st.sidebar.selectbox("페이지 선택", pages)

if page == "홈":
    st.title("KBO리그 시뮬레이션")
    st.markdown("""
    # KBO리그 시뮬레이션
    이 앱은 MLB 데이터와 KBO 데이터를 기반으로 팀 성적을 예측하는 시뮬레이션 앱입니다.

    ## 사용된 주요 야구 지표
    - **OPS (On-base Plus Slugging)**: 타자의 출루율과 장타율을 합한 값
    - **SLG (Slugging Percentage)**: 타자의 장타율
    - **OBP (On-base Percentage)**: 타자의 출루율
    - **ERA (Earned Run Average)**: 투수의 평균 자책점
    - **WHIP (Walks plus Hits per Inning Pitched)**: 이닝당 출루 허용률

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

elif page == "선수 성적 일람":
    st.title("선수 성적 일람")
    
    # 제공된 CSV 파일 경로
    file_path = 'data/player_data_2013~2023.csv'
    player_data = pd.read_csv(file_path)
    
    st.dataframe(player_data)

    # 선수 성적 요약 정보 제공
    summary = player_data.describe()
    st.write("선수 성적 요약")
    st.dataframe(summary)

    # 각 팀별 성적 요약
    teams = player_data['Team'].unique()
    for team in teams:
        st.write(f"팀: {team}")
        team_data = player_data[player_data['Team'] == team]
        st.dataframe(team_data)

# 팀 라인업 편집 페이지
elif page == "팀 편집":
    st.title("팀 라인업 편집")
    
    # 제공된 CSV 파일 경로
    file_path = 'data/player_data_2013~2023.csv'
    player_data = pd.read_csv(file_path)
    
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
    
    # 포지션별 선수 입력 필드
    for pos in positions:
        st.write(f"### {pos}")
        available_players = player_data.dropna(subset=['OPS', 'SLG', 'OBP']) if pos not in ['sp', 'rp'] else player_data.dropna(subset=['ERA', 'WHIP'])
        for idx, player in enumerate(team_lineup[pos]):
            current_player = player[0] if player[0] in available_players['Player'].values else '선수 선택'
            years = sorted(player_data[player_data['Player'] == current_player]['Year'].unique(), reverse=True) if current_player != '선수 선택' else []
            current_year = player[1] if player[1] in years else (years[0] if years else '연도 선택')
            
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                selected_player = st.selectbox(f"{pos} 선수 {idx+1}", ['선수 선택'] + list(available_players['Player'].values), index=0 if current_player == '선수 선택' else available_players['Player'].values.tolist().index(current_player) + 1, key=f"{pos}_{idx}_player")
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
                        selected_year = st.selectbox(f"{pos} 연도 {idx+1}", years, index=0 if current_year == '연도 선택' else years.index(current_year), key=f"{pos}_{idx}_year")
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
        if st.button(f"{pos} 새로운 선수 추가", key=f"add_{pos}"):
            team_lineup[pos].append(['선수 선택', '연도 선택'])
            st.experimental_rerun()

    # 선수 이름이 "선수 선택"인 칸을 모두 삭제
    for pos in positions:
        team_lineup[pos] = [player for player in team_lineup[pos] if player[0] != '선수 선택']

    st.write("편집된 팀 라인업")
    st.write(dict(team_lineup))
    
    # 변경 사항 저장
    st.session_state.lineup_state[selected_team] = team_lineup
    
    # Save updated lineup to a JSON file or display as JSON string
    st.json(dict(team_lineup))

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
