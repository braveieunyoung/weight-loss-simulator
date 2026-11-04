import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="체중 감량 모의 투자 시뮬레이터",
    page_icon="📈",
    layout="centered"
)

# 모델 로드 (캐싱을 이용해 한 번만 로드)
@st.cache_resource
def load_models():
    lr = joblib.load('lr_model.pkl')
    rf = joblib.load('rf_clf.pkl')
    return lr, rf

lr_model, rf_clf = load_models()

# UI 타이틀
st.title("📈 내 몸무게 가치 상승 (체중 감량 모의 투자) 프로젝트")
st.markdown("신체 데이터를 입력하고 **감량 포트폴리오(시나리오)**를 설계해 보세요!")

st.divider()

# 사이드바 또는 메인 입력 폼
st.subheader("🛠️ 현재 신체 자산(스펙) 입력")
col1, col2 = st.columns(2)

with col1:
    height = st.number_input("키 (cm)", min_value=100.0, max_value=250.0, value=180.0, step=0.1)
    chest = st.number_input("가슴둘레 (cm)", min_value=50.0, max_value=150.0, value=95.0, step=0.1)

with col2:
    waist = st.number_input("허리둘레 (cm)", min_value=50.0, max_value=150.0, value=95.0, step=0.1)
    target_weight = st.number_input("목표 체중 (kg)", min_value=30.0, max_value=150.0, value=62.0, step=0.1)

st.markdown("")
if st.button("🚀 감량 시나리오 분석 및 투자 설계", use_container_width=True):
    # 1. 선형 회귀 기반 현재 몸무게 예측
    input_df = pd.DataFrame([[height, chest, waist]], columns=['height', 'chest', 'waist'])
    predicted_weight = float(lr_model.predict(input_df)[0])

    # 2. 현재 상태 비만 단계 예측
    input_current = pd.DataFrame([[height, chest, waist, predicted_weight]], columns=['height', 'chest', 'waist', 'weight'])
    pred_obesity_rf = str(rf_clf.predict(input_current)[0])

    # 3. 감량 목표 계산
    weight_to_lose = predicted_weight - target_weight

    st.divider()
    st.subheader("📊 [현재 자산 분석 리포트]")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("현재 예측 몸무게", f"{predicted_weight:.1f} kg")
    m2.metric("목표 몸무게", f"{target_weight:.1f} kg")
    m3.metric("현재 비만 단계", pred_obesity_rf, delta_color="inverse")

    if weight_to_lose <= 0:
        st.success("🎉 축하합니다! 이미 목표 체중 이하이거나 추가 감량이 필요 없는 상태입니다.")
    else:
        st.info(f"🎯 **총 감량 목표 (청산해야 할 초과 체중):** -{weight_to_lose:.1f} kg")

        # 선형 회귀 계수 추출
        coef_chest = lr_model.coef_[1]
        coef_waist = lr_model.coef_[2]

        # 시나리오 1: 허리 집중 감량
        waist_reduction_only = weight_to_lose / coef_waist
        target_waist_only = waist - waist_reduction_only
        input_target_1 = pd.DataFrame([[height, chest, target_waist_only, target_weight]], columns=['height', 'chest', 'waist', 'weight'])
        pred_obesity_target_1 = str(rf_clf.predict(input_target_1)[0])

        # 시나리오 2: 허리 70% + 가슴 30% 균형 감량
        target_weight_loss_waist = weight_to_lose * 0.7
        target_weight_loss_chest = weight_to_lose * 0.3
        waist_reduction_balanced = target_weight_loss_waist / coef_waist
        chest_reduction_balanced = target_weight_loss_chest / coef_chest
        target_waist_balanced = waist - waist_reduction_balanced
        target_chest_balanced = chest - chest_reduction_balanced
        input_target_2 = pd.DataFrame([[height, target_chest_balanced, target_waist_balanced, target_weight]], columns=['height', 'chest', 'waist', 'weight'])
        pred_obesity_target_2 = str(rf_clf.predict(input_target_2)[0])

        # 화면에 시나리오 카드 출력
        st.markdown("---")
        st.subheader("💡 감량 포트폴리오 (시나리오) 추천")

        tab1, tab2 = st.tabs(["📉 [종목 1] 허리둘레 집중 매집형", "⚖️ [종목 2] 전신 균형 밸런스형"])

        with tab1:
            st.markdown("### 허리둘레 집중 감량 전략")
            st.write(f"- **허리둘레 목표:** {waist}cm ➔ **{target_waist_only:.1f}cm** (약 **{waist_reduction_only:.1f}cm** 감량)")
            st.write(f"- **비만 단계 변화:** {pred_obesity_rf} ➔ **{pred_obesity_target_1}**")
            st.success("💪 **추천 트레이닝 루틴:** 유산소(러닝, 인터벌) + 복근/코어 운동(플랭크, 크런치, 레그레이즈)")

        with tab2:
            st.markdown("### 전신 균형 감량 포트폴리오 (허리 70% + 상체 30%)")
            st.write(f"- **허리둘레 목표:** {waist}cm ➔ **{target_waist_balanced:.1f}cm** (약 **{waist_reduction_balanced:.1f}cm** 감량)")
            st.write(f"- **가슴둘레 목표:** {chest}cm ➔ **{target_chest_balanced:.1f}cm** (약 **{chest_reduction_balanced:.1f}cm** 감량)")
            st.write(f"- **비만 단계 변화:** {pred_obesity_rf} ➔ **{pred_obesity_target_2}**")
            st.success("💪 **추천 트레이닝 루틴:** 전신 유산소 + 상체 근력 운동(푸시업, 덤벨 숄더프레스, 렛풀다운)으로 근선명도 유지")
