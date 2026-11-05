import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# 페이지 설정
st.set_page_config(
    page_title="체중 감량 모의 투자 시뮬레이터",
    page_icon="📈",
    layout="wide"
)

# 1. 앱이 실행될 때 데이터를 읽고 모델을 학습시키는 함수 (캐싱 적용)
@st.cache_resource
def train_and_load_models():
    df = pd.read_csv('신체검사.csv')
    
    # 전처리
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['weight'] != 0]
    df = df.dropna(subset=['chest', 'height', 'waist', 'weight'])
    
    # BMI 계산 및 비만도 범주 정의
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    
    def categorize_bmi(val):
        if val < 18.5:
            return '저체중'
        elif val < 23.0:
            return '정상'
        elif val < 25.0:
            return '과체중'
        else:
            return '비만'

    df['BMI'] = bmi.apply(categorize_bmi)
    
    # 독립변수(X)와 종속변수(y) 분리 - 회귀용 (몸무게 예측)
    X_reg = df[['height', 'chest', 'waist']]
    y_reg = df['weight']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_r, y_train_r)
    
    # 분류용 (BMI 비만도 예측)
    X_clf = df[['height', 'chest', 'waist', 'weight']]
    y_clf = df['BMI']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_c, y_train_c)
    
    return lr_model, rf_clf

# 모델 로드
lr_model, rf_clf = train_and_load_models()

# 2. 웹 UI 구성
st.title("체중 감량 프로젝트")
st.markdown("신체 데이터를 입력하고 감량 시나리오를 한눈에 비교해 보세요.")

st.divider()

st.subheader("현재 신체 치수 입력")
col1, col2, col3, col4 = st.columns(4)

with col1:
    h = st.number_input("키 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
with col2:
    c = st.number_input("가슴둘레 (cm)", min_value=50.0, max_value=150.0, value=90.0, step=0.1)
with col3:
    w = st.number_input("허리둘레 (cm)", min_value=50.0, max_value=150.0, value=80.0, step=0.1)
with col4:
    w2 = st.number_input("목표 몸무게 (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.1)

st.markdown("")

# 3. 분석 버튼 클릭 시 실행
if st.button("분석 시작하기", use_container_width=True):
    # 현재 몸무게 예측
    input_df = pd.DataFrame([[h, c, w]], columns=['height', 'chest', 'waist'])
    predicted_weight = lr_model.predict(input_df)[0]

    # 현재 비만 단계 예측
    input_current = pd.DataFrame([[h, c, w, predicted_weight]], columns=['height', 'chest', 'waist', 'weight'])
    pred_obesity_rf = rf_clf.predict(input_current)[0]

    weight_to_lose = predicted_weight - w2

    st.divider()
    st.subheader("분석 리포트")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 몸무게", f"{predicted_weight:.1f} kg")
    m2.metric("목표 몸무게", f"{w2:.1f} kg")
    m3.metric("현재 비만 단계", pred_obesity_rf)

    if weight_to_lose <= 0:
        st.success("이미 목표 체중 이하이거나 추가 감량이 필요 없는 상태입니다.")
    else:
        st.info(f"감량 목표: -{weight_to_lose:.1f} kg")

        # 회귀 계수 추출
        coef_chest = lr_model.coef_[1]
        coef_waist = lr_model.coef_[2]

        # 시나리오 1: 허리 집중 감량
        waist_reduction_only = weight_to_lose / coef_waist
        target_waist_only = w - waist_reduction_only

        input_target_1 = pd.DataFrame([[h, c, target_waist_only, w2]], columns=['height', 'chest', 'waist', 'weight'])
        pred_obesity_target_1 = rf_clf.predict(input_target_1)[0]

        # 시나리오 2: 허리 70% + 가슴 30% 균형 감량
        target_weight_loss_waist = weight_to_lose * 0.7
        target_weight_loss_chest = weight_to_lose * 0.3

        waist_reduction_balanced = target_weight_loss_waist / coef_waist
        chest_reduction_balanced = target_weight_loss_chest / coef_chest

        target_waist_balanced = w - waist_reduction_balanced
        target_chest_balanced = c - chest_reduction_balanced

        input_target_2 = pd.DataFrame([[h, target_chest_balanced, target_waist_balanced, w2]], columns=['height', 'chest', 'waist', 'weight'])
        pred_obesity_target_2 = rf_clf.predict(input_target_2)[0]

        st.markdown("---")
        st.subheader("감량 시나리오 추천 비교")

        # 화면을 좌우 2개의 컬럼으로 분할하여 시나리오를 동시에 표시
        scenario_col1, scenario_col2 = st.columns(2)

        with scenario_col1:
            with st.container(border=True):
                st.markdown("### 허리둘레 집중형")
                st.write(f"- 허리둘레 목표: {w}cm ➔ {target_waist_only:.1f}cm")
                st.write(f"  (약 {waist_reduction_only:.1f}cm 감량 필요)")
                st.write(f"- 비만단계 변화: {pred_obesity_rf} ➔ {pred_obesity_target_1}")
                st.write("추천 운동: 유산소(러닝, 인터벌 유산소) + 복근/코어 운동(플랭크, 크런치, 레그레이즈)")

        with scenario_col2:
            with st.container(border=True):
                st.markdown("### 전신 균형 밸런스형")
                st.write(f"- 허리둘레 목표: {w}cm ➔ {target_waist_balanced:.1f}cm")
                st.write(f"  (약 {waist_reduction_balanced:.1f}cm 감량 필요)")
                st.write(f"- 가슴둘레 목표: {c}cm ➔ {target_chest_balanced:.1f}cm")
                st.write(f"  (약 {chest_reduction_balanced:.1f}cm 감량 필요)")
                st.write(f"- 비만단계 변화: {pred_obesity_rf} ➔ {pred_obesity_target_2}")
                st.write("추천 운동: 전신 유산소 + 상체 근력 운동(푸시업, 덤벨 숄더프레스, 렛풀다운)")
