import streamlit as st
from st_screen_stats import ScreenData
import asyncio
import json
from openai import AsyncOpenAI, OpenAI
import os
import aiohttp
import requests
import re
from dotenv import load_dotenv
from google.auth import default
import google.auth.transport.requests

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from transformers import AutoTokenizer
from PIL import Image
import base64

# --- 페이지 기본 설정 ---
st.set_page_config(
    layout="wide", 
    page_title="모델 지문 생성 뷰어",
    initial_sidebar_state="collapsed"
)

# --- 로고를 base64로 인코딩하여 HTML에 직접 삽입 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 로고 이미지를 base64로 인코딩
img_base64 = get_base64_of_bin_file("logo_kangnam_202111.png")


# --- 사용자 정의 CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@400;700;800&display=swap');

h3 {
    font-family: 'Nanum Myeongjo', serif !important;
    font-weight: 700;
}
.passage-container {
    height: 80vh;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 15px;
}
.passage-font {
    border: 0.5px solid black;
    border-radius: 0px;
    padding: 10px;
    margin-bottom: 20px;
    font-family: 'Nanum Myeongjo', serif !important;
    line-height: 1.7;
    letter-spacing: -0.01em;
    font-weight: 500;
}
.passage-font p {
    text-indent: 1em; /* 각 문단의 첫 줄 들여쓰기 */
    margin-bottom: 0em;
}
.passage-font-no-border {
    padding: 10px;
    margin-bottom: 20px;
    font-family: 'Nanum Myeongjo', serif !important;
    line-height: 1.7;
    letter-spacing: -0.01em;
    font-weight: 500;
}
.passage-font-no-border p {
    text-indent: 1em; /* 각 문단의 첫 줄 들여쓰기 */
    margin-bottom: 0em;
}
.question-font {
    font-family: 'Nanum Myeongjo', serif !important;
    line-height: 1.7em;
    letter-spacing: -0.01em;
    font-weight: 500;
    margin-bottom: 1.5em;
}
.final-response-container {
    height: 600px;  /* 기본값, 동적으로 덮어씌워짐 */
    overflow-y: auto;
    border: 0.5px solid black;
    border-radius: 0px;
    padding: 5px;
    background-color: white;
}
</style>
""", unsafe_allow_html=True)


# --- 설정값 ---
DATASET_PATH = "Gemini-sft-09-07-val.jsonl"
# Vertex AI 조정된 모델 설정
ENDPOINT_ID = "4075215603537805312"  # 사용자 지정 엔드포인트 ID (ksat-exp-09-06-flash)
PROJECT_ID = "gen-lang-client-0921402604"  # GCP 프로젝트 ID  
LOCATION = "us-central1"  # 모델이 배포된 리전
MODEL_ID = "6275144856671092736"  # 실제 모델 ID (ksat-exp-09-06-flash)

# 기존 OpenAI 모델들 (참고용)
#MODEL_NAME = "ft:gpt-4.1-2025-04-14:ksat-agent:ksat-exp-09-06-large:CCMOwou1"
EXPERT_MODEL_NAME = "gemini-2.5-flash"

EXPERT_PROMPT = """
당신은 작가 모델에게 수능 지문을 작성하기 위해 필요한 정보를 제공하는 전문가 모델입니다.
작가 모델이 요청하는 정보를 아래의 지침에 따라 제공해 주세요.
1. 특정 개념이나 인물의 주장을 여러 개 제시할 때에는 벙렬적으로 나열하지 말고 각 개념 또는 주장의 공통점 및 차이점이 발생하는 대립점을 명확히 하여 정보를 제공해 주세요.
2. 과학적/경제적 원리 또는 기술의 작동 원리를 제시할 때에는 원리를 피상적/광범위하게 나열하기 말고, 미시적이고 깊이 있게 설명하여 정보를 제공해 주세요.
3. 법적 규정이나 제도의 작동 원리를 제시할 때에는 규정이나 제도를 줄줄이 나열하기보다는, 해당 규정이 등장한 배경과 목적(또는 해결하고자 하는 문제), 규정 또는 제도가 해당 목적 달성을 위해 작동하는 원리에 초점을 맞춰 설명해 주세요.
4. 작가 모델이 다소 광범위한 주제에 대해 질문한다면, 작가 모델이 지문에 포함할 내용을 탐색하고 있는 것입니다. 다음의 서사 구조 중 하나를 살려 입체적으로 정보를 제공해 주세요.
    - 문제 발생 및 해결 구조 : 문제가 발생하는 원리와 그 원리를 해결하기 위한 수단과 방법을 제시하고, 그 원리를 상세하게 설명합니다. 추가로 해결법의 한계와 그 한계를 극복하기 위한 다른 방법을 제시하는 것도 바람직합니다.
    - 다양한 견해 비교 구조 : 하나의 화제에 대한 여러 인물이나 학파의 관점을 순차적으로 제시하고, 그들의 해석과 주장의 차이점, 때로는 서로의 견해에 대한 비판과 그에 대한 반박을 명확하게 드러냅니다. 이 구조는 각 견해의 핵심 내용을 정확히 파악하고 서로 비교/대조하는 능력을 평가합니다.
    - 개념(또는 조건) 및 적용 사례 구조 : 특정 개념이나 제도를 정의한 뒤, 이와 관련된 법률이나 규칙의 구체적인 조항과 조건을 상세히 제시합니다. 또한 해당 구조가 적용될 수 있는 구체적인 사례를 제시해도 좋습니다. 이 구조는 지문의 정보를 구체적인 사례에 적용하는 능력을 평가합니다.
5. 수능은 배경지식이 아닌 논리적 규칙을 이해하고 적용하는 시험입니다. 개념의 양과 다양성보다는 조건, 인과, 대립, 분기, 위계 등 논리적 규칙을 명확히 하여 정보를 제공해 주세요.
6. 수식을 통한 설명보다는, 언어를 활용하여 논리적으로 설명해 주세요.
7. 배경 지식 수준은 다음과 같이 고려하여 정보를 제공해 주세요.
    - 수학: 사칙연산과 거듭제곱 정도의 기초적인 수학 지식만을 갖춘 독자를 전제로 설명해야 합니다.
    - 과학: 힘, 속도, 거리, 분자, 원자, 바이러스, 미생물 등 기초적인 개념만을 갖춘 독자를 전제해야 합니다.
    - 사회: 화폐, 경기, 통화, 채권, 민법, 국회, 보도 등 기초적인 사회 용어를 숙지한 독자를 전제합니다. 
8. 해당 도메인의 전문/특수 용어 대신, 일상적인 용어와 사례를 사용해 주세요.
9. 정보는 요청에 충실하되 간결하게 핵심만 제공해 주세요.
"""

# --- Vertex AI 조정된 모델 호출 함수 ---
def get_vertex_ai_credentials():
    """Vertex AI 인증 토큰을 가져옵니다."""
    try:
        # 1. Streamlit Cloud Secrets에서 JSON 문자열로 제공되는 서비스 계정 키 처리
        if hasattr(st, 'secrets') and "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
            import json
            from google.oauth2 import service_account
            
            try:
                creds_json = json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
                credentials = service_account.Credentials.from_service_account_info(
                    creds_json, 
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                credentials.refresh(google.auth.transport.requests.Request())
                return credentials.token
            except Exception as e:
                st.error(f"Streamlit Secrets 인증 실패: {e}")
        
        # 2. 로컬 환경에서 gcloud CLI 인증 시도
        original_creds = os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
        
        try:
            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(google.auth.transport.requests.Request())
            return credentials.token
        finally:
            if original_creds:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_creds
                
    except Exception as e:
        # 3. gcloud 인증 실패 시, 환경변수 인증 재시도
        if 'original_creds' in locals() and original_creds and os.path.exists(original_creds):
            try:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = original_creds
                credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
                credentials.refresh(google.auth.transport.requests.Request())
                return credentials.token
            except Exception as e2:
                pass
        
        # 인증 방법 안내
        st.error(f"Vertex AI 인증 실패: {e}")
        with st.expander("🔧 인증 설정 방법", expanded=True):
            st.markdown("""
            **Streamlit Cloud 배포 시:**
            1. Settings > Secrets에서 다음 추가:
            ```toml
            GOOGLE_APPLICATION_CREDENTIALS_JSON = '''
            {
              "type": "service_account",
              "project_id": "your-project-id",
              ...전체 서비스 계정 키 JSON...
            }
            '''
            ```
            
            **로컬 개발 시:**
            ```bash
            gcloud auth application-default login
            ```
            
            **또는 서비스 계정 키 파일:**
            ```bash
            export GOOGLE_APPLICATION_CREDENTIALS="키파일경로.json"
            ```
            """)
        return None

async def call_vertex_ai_endpoint(endpoint_id: str, project_id: str, location: str, messages: list, temperature: float = 0.7):
    """Vertex AI 조정된 모델 엔드포인트에 직접 요청을 보냅니다."""
    access_token = get_vertex_ai_credentials()
    if not access_token:
        return None
    
    # Vertex AI 엔드포인트 URL
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Gemini API 형식으로 메시지 변환
    contents = []
    system_instruction = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": msg["content"]}]
            })
        elif msg["role"] == "assistant":
            contents.append({
                "role": "model", 
                "parts": [{"text": msg["content"]}]
            })
    
    # 요청 페이로드
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "candidates" in result and result["candidates"]:
                        content = result["candidates"][0]["content"]["parts"][0]["text"]
                        return content
                    else:
                        return "[error] No response from model"
                else:
                    error_text = await response.text()
                    return f"[error] HTTP {response.status}: {error_text}"
    except Exception as e:
        return f"[error] Request failed: {e}"

def call_vertex_ai_endpoint_sync(endpoint_id: str, project_id: str, location: str, messages: list, temperature: float = 0.7):
    """동기식 Vertex AI 조정된 모델 호출"""
    access_token = get_vertex_ai_credentials()
    if not access_token:
        return "[error] Authentication failed"
    
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:generateContent"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Gemini API 형식으로 메시지 변환
    contents = []
    system_instruction = None
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": msg["content"]}]
            })
        elif msg["role"] == "assistant":
            contents.append({
                "role": "model", 
                "parts": [{"text": msg["content"]}]
            })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 8192,
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                content = result["candidates"][0]["content"]["parts"][0]["text"]
                return content
            else:
                return "[error] No response from model"
        else:
            return f"[error] HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return f"[error] Request failed: {e}"

# --- 일반 Gemini API 클라이언트 (Expert용) ---
def create_openai_client(use_vertex_ai: bool = False, project_id: str = "", location: str = "") -> AsyncOpenAI:
    """Expert용 일반 Gemini API 클라이언트를 생성합니다."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        st.error("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        st.info("💡 Google AI Studio에서 API 키를 생성하고 환경변수로 설정하세요: https://aistudio.google.com/app/apikey")
        return AsyncOpenAI()

def create_sync_openai_client(use_vertex_ai: bool = False, project_id: str = "", location: str = "") -> OpenAI:
    """Expert용 동기식 일반 Gemini API 클라이언트를 생성합니다."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        return OpenAI()

# --- 도우미 함수 ---
@st.cache_data
def get_dataset_info(path):
    """데이터셋 파일의 총 샘플 수를 반환합니다."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0

@st.cache_data
def load_sample(index):
    """지정된 인덱스의 데이터셋 샘플을 로드합니다."""
    try:
        with open(DATASET_PATH, "r", encoding='utf-8') as f:
            line = f.readlines()[index]
            data = json.loads(line)
            
            # Gemini 형식에서 시스템 프롬프트, 사용자 프롬프트, 기대 응답 추출
            system_prompt = ""
            user_prompt = ""
            expected_response = ""
            
            # systemInstruction에서 시스템 프롬프트 추출
            if "systemInstruction" in data:
                system_instruction = data["systemInstruction"]
                if isinstance(system_instruction, dict) and "parts" in system_instruction:
                    if system_instruction["parts"] and "text" in system_instruction["parts"][0]:
                        system_prompt = system_instruction["parts"][0]["text"]
            
            # contents에서 user와 model 메시지 추출
            if "contents" in data:
                contents = data["contents"]
                for content in contents:
                    role = content.get("role", "")
                    parts = content.get("parts", [])
                    
                    if role == "user" and parts and "text" in parts[0]:
                        if not user_prompt:  # 첫 번째 user 메시지를 사용
                            user_prompt = parts[0]["text"]
                    
                    elif role == "model" and parts and "text" in parts[0]:
                        expected_response_raw = parts[0]["text"]
                        # <passage> 태그 내용 추출 또는 전체 내용 사용
                        if "<passage>" in expected_response_raw and "</passage>" in expected_response_raw:
                            start = expected_response_raw.find("<passage>") + len("<passage>")
                            end = expected_response_raw.find("</passage>")
                            expected_response = expected_response_raw[start:end].strip()
                        else:
                            # </think> 이후 내용 추출
                            if "</think>" in expected_response_raw:
                                expected_response = expected_response_raw.split("</think>", 1)[1].strip()
                            else:
                                expected_response = expected_response_raw.strip()
            
            return system_prompt, user_prompt, expected_response
    except Exception as e:
        st.error(f"데이터셋 로딩 오류: {e}")
        return None, None, None

def format_text_to_html(text: str) -> str:
    """텍스트의 줄바꿈을 HTML 단락(<p>)으로 변환합니다."""
    paragraphs = text.strip().split('\n')
    html_paragraphs = [f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()]
    return "".join(html_paragraphs)

def parse_prompt_structure(user_prompt: str) -> tuple[str, str, str]:
    """사용자 프롬프트를 파싱하여 분야, 유형, 주제를 추출합니다."""
    try:
        lines = user_prompt.strip().split('\n')
        field_info = ""
        type_info = ""
        topic_info = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("분야:"):
                field_info = line.replace("분야:", "").strip()
            elif line.startswith("유형:"):
                type_info = line.replace("유형:", "").strip()
            elif line.startswith("주제:"):
                topic_info = line.replace("주제:", "").strip()
        
        return field_info, type_info, topic_info
    except Exception:
        return "파싱 실패", "파싱 실패", "파싱 실패"

def format_prompt_from_components(field: str, type_info: str, topic: str) -> str:
    """분야, 유형, 주제를 결합하여 프롬프트 형식으로 변환합니다."""
    return f"분야: {field}\n유형: {type_info}\n주제: {topic}"


# --- 사이드바 (높이 자동 감지) ---
with st.sidebar:
    # 스트리밍 중이 아닐 때만 높이 감지
    if not st.session_state.get("is_streaming", False):
        try:
            with st.container(border=False, height=1):
                screen_data = ScreenData()
                stats = screen_data.st_screen_data()  # 컴포넌트 로딩 및 값 가져오기

            if stats and "innerHeight" in stats:
                height = stats.get("innerHeight")
                if height is not None and isinstance(height, (int, float)) and height > 0:
                    # 세션 상태에 최신 높이 저장/업데이트 (현재 높이와 다를 경우에만 업데이트)
                    if st.session_state.get("viewport_height") != height:
                        st.session_state.viewport_height = height
                else:
                    pass
        except Exception as e:
            pass
            # 오류 발생 시에도 세션 상태에 viewport_height가 없으면 기본값 설정
            if "viewport_height" not in st.session_state:
                st.session_state.viewport_height = 800  # 기본값 설정

    # 높이 감지는 하지만 표시하지 않음

# --- 메인 페이지 로고 & 타이틀 (상단) ---
st.markdown(f"""
<div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 20px;">
    <img src="data:image/png;base64,{img_base64}" 
         style="width: 170px; height: auto; pointer-events: none; user-select: none; margin-right: 30px;" 
         alt="강남대성수능연구소 로고">
    <div class="passage-font-no-border" style="margin: 0; padding: 0; font-size: 40px; font-weight: 700;">
        AI Model Preview
    </div>
</div>
""", unsafe_allow_html=True)

# --- 동적 높이 계산 ---
# 기본 높이 600px, 감지된 높이가 있으면 적절히 조절
default_height = 600
if "viewport_height" in st.session_state:
    # 화면 높이의 70%를 컨테이너 높이로 사용 (헤더, 여백 등을 고려)
    dynamic_height = int(st.session_state.viewport_height * 0.8)
    # 최소 400px, 최대 1000px로 제한
    container_height = max(400, min(dynamic_height, 1000))
else:
    container_height = default_height

# --- 동적 CSS 적용 ---
st.markdown(f"""
<style>
.final-response-container {{
    height: {container_height}px !important;
}}
</style>
""", unsafe_allow_html=True)


# --- 3컬럼 레이아웃 ---
col1, col2, col3 = st.columns([1, 1, 1])

# 첫 번째 컬럼: 설정 및 입력
with col1:
    st.markdown("#### 1. 설정 및 입력")
    with st.container(border=True, height=container_height):        
        # 모델 선택 섹션
        with st.container(border=True):
            st.markdown("**AI 모델 선택**")
            model_name = st.selectbox("모델명", ["KSAT Psg Flash (Preview 0908)"], index=0)
        
        # 입력 프롬프트 섹션
        with st.container(border=True):
            st.markdown("**입력 프롬프트**")
            tab1, tab2 = st.tabs(["Preset", "Custom"])
            
            with tab1:
                # 데이터셋 샘플 선택
                total_samples = get_dataset_info(DATASET_PATH)
                if total_samples > 0:
                    dataset_index = st.selectbox(
                        "검증 데이터셋 샘플",
                        options=range(total_samples),
                        format_func=lambda x: f"Sample #{x}",
                        index=0
                    )
                    
                    # 선택된 샘플 로드 및 파싱
                    system_prompt, user_prompt, expected_response = load_sample(dataset_index)
                    if user_prompt:
                        field_info, type_info, topic_info = parse_prompt_structure(user_prompt)
                        
                        # 파싱된 정보 표시 (읽기 전용)
                        st.text_input("분야", value=field_info, disabled=True, key="preset_field_display")
                        st.text_input("유형", value=type_info, disabled=True, key="preset_type_display")
                        st.text_area("주제", value=topic_info, disabled=True, height=100, key="preset_topic_display")
                        
                        # 원본 지문 익스팬더
                        with st.expander("원본 지문", expanded=False):
                            if expected_response:
                                st.markdown(f'<div class="passage-font">{format_text_to_html(expected_response)}</div>', unsafe_allow_html=True)
                        
                        # Preset 탭 실행 버튼
                        preset_run_button = st.button("지문 생성", type="primary", use_container_width=True, key="preset_run")
                    else:
                        st.error("선택된 샘플의 프롬프트를 파싱할 수 없습니다.")
                        preset_run_button = False
                else:
                    st.error("데이터셋 파일을 찾을 수 없습니다.")
                    preset_run_button = False
            
            with tab2:
                # 커스텀 입력
                # 분야 선택 (대분야-소분야 구조)
                custom_major_field = st.selectbox("대분야", ["인문사회", "과학기술"], index=0, key="custom_major_field")
                
                if custom_major_field == "인문사회":
                    custom_minor_field = st.selectbox("소분야", ["인문", "예술", "법", "경제"], index=0, key="custom_minor_field")
                else:  # 과학기술
                    custom_minor_field = st.selectbox("소분야", ["과학", "기술"], index=0, key="custom_minor_field")
                
                custom_type = st.selectbox("유형", ["단일형", "(가), (나) 분리형"], index=0, key="custom_type")
                custom_topic = st.text_area("주제", value="", height=100, 
                                          placeholder="여기에 원하는 주제를 입력해주세요.",
                                          key="custom_topic")
                
                # Custom 탭 실행 버튼
                custom_run_button = st.button("지문 생성", type="primary", use_container_width=True, key="custom_run")
        
        # Temperature 섹션 (프롬프트 아래로 이동)
        with st.container(border=True):
            st.markdown("**Temperature**")
            st.markdown("다양성을 조절하는 파라미터입니다. 높을수록 지문의 내용이 다채롭지만, 다소 산만하고 일관성이 떨어질 수 있습니다.\n(기본값: 0.85)")
            
            # Temperature 초기값 설정
            if "temperature" not in st.session_state:
                st.session_state.temperature = 0.85
                
            temperature = st.slider("", 0.5, 1.2, st.session_state.temperature, 0.05, key="temp_slider")
            
            # slider 값이 변경되면 session_state 업데이트
            st.session_state.temperature = temperature

# 두 번째 컬럼: Reasoning & Expert Response
with col2:
    st.markdown("#### 2. 모델 사고 과정")
    with st.container(border=True, height=container_height):
        reasoning_placeholder = st.empty()
        reasoning_placeholder.info("AI 모델의 사고 과정이 여기에 표시됩니다.")

# 세 번째 컬럼: Final Response
with col3:
    st.markdown("#### 3. 최종 지문")
    final_placeholder = st.empty()
    # 커스텀 CSS 컨테이너로 초기 상태 표시
    final_placeholder.markdown('''
    <div class="final-response-container">
        <p style="color: #666; text-align: center; margin-top: 250px;">최종 지문이 여기에 표시됩니다.</p>
    </div>
    ''', unsafe_allow_html=True)


# --- 도구 함수 (전문가 호출) ---
async def execute_request_for_expert(input_text: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "[expert_error] Missing GOOGLE_API_KEY"

    def _call_sync() -> str:
        try:
            # Expert 모델은 항상 Google API 사용 (Gemini)
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            resp = client.chat.completions.create(
                model=EXPERT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": EXPERT_PROMPT},
                    {"role": "user", "content": input_text},
                ],
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[expert_error] {e}"

    result = await asyncio.to_thread(_call_sync)
    return result or "[expert_empty]"


# --- 텍스트에서 expert 태그 파싱 함수 ---
import re

def parse_expert_calls(text: str) -> tuple[str, list[str]]:
    """
    텍스트에서 <expert>...</expert> 또는 <expertcall>...</expertcall> 태그를 찾아서 파싱합니다.
    
    Returns:
        (cleaned_text, expert_calls): 태그가 제거된 텍스트와 전문가 호출 리스트
    """
    expert_calls = []
    
    # <expert>...</expert> 패턴 찾기 
    expert_pattern = r'<expert>(.*?)</expert>'
    expert_matches = re.findall(expert_pattern, text, re.DOTALL)
    
    # <expertcall>...</expertcall> 패턴 찾기
    expertcall_pattern = r'<expertcall>(.*?)</expertcall>'
    expertcall_matches = re.findall(expertcall_pattern, text, re.DOTALL)
    
    # 모든 매칭 결과를 합치기
    for match in expert_matches:
        expert_calls.append(match.strip())
    
    for match in expertcall_matches:
        expert_calls.append(match.strip())
    
    # 두 패턴 모두 제거한 텍스트 반환
    cleaned_text = re.sub(expert_pattern, '', text, flags=re.DOTALL)
    cleaned_text = re.sub(expertcall_pattern, '', cleaned_text, flags=re.DOTALL).strip()
    
    return cleaned_text, expert_calls

# --- 새로운 스트리밍 함수 (Vertex AI 조정된 모델 기반) ---
async def run_vertex_ai_flow_streaming(endpoint_id: str, project_id: str, location: str, system_prompt: str, user_prompt: str, temperature: float):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    round_idx = 0
    final_text = ""

    while True:
        round_idx += 1
        if round_idx > 30:
            break

        try:
            # Vertex AI 조정된 모델 호출
            content = await call_vertex_ai_endpoint(
                endpoint_id=endpoint_id,
                project_id=project_id,
                location=location,
                messages=messages,
                temperature=temperature
            )
            
            if not content or content.startswith("[error]"):
                yield {"type": "think", "content": f"Model error: {content}"}
                break
            
            # assistant 메시지를 히스토리에 추가
            messages.append({"role": "assistant", "content": content})
            
            # <expert> 태그 파싱
            cleaned_text, expert_calls = parse_expert_calls(content)
            
            # <passage> 태그 확인 - 최종 응답 여부 판단
            has_passage = "<passage>" in cleaned_text and "</passage>" in cleaned_text
            
            if has_passage:
                # <passage> 태그에서 최종 지문 추출
                start_tag = "<passage>"
                end_tag = "</passage>"
                start_idx = cleaned_text.find(start_tag)
                end_idx = cleaned_text.find(end_tag)
                
                if start_idx != -1 and end_idx != -1:
                    # <passage> 이전 부분 (thinking/reasoning)
                    before_passage = cleaned_text[:start_idx].strip()
                    # <passage> 내용 (최종 지문)
                    passage_content = cleaned_text[start_idx + len(start_tag):end_idx].strip()
                    
                    # 이전 부분이 있으면 thinking으로 출력
                    if before_passage:
                        yield {"type": "think", "content": before_passage}
                    
                    remaining_text = passage_content
                    is_final_response = True
                else:
                    remaining_text = cleaned_text
                    is_final_response = False
            else:
                # <passage> 태그가 없으면 일반 텍스트로 처리
                remaining_text = cleaned_text
                is_final_response = False
            
            # expert 호출이 있는 경우
            if expert_calls:
                # expert 호출 전에 중간 메시지가 있으면 thinking으로 출력
                if remaining_text and not has_passage:
                    yield {"type": "think", "content": remaining_text}
                
                for expert_input in expert_calls:
                    # 전문가 질의 시작 이벤트 (질의 내용 먼저 표시)
                    yield {"type": "tool_start", "input": expert_input}
                    
                    # 전문가 함수 호출 (일반 Gemini API 사용)
                    expert_result = await execute_request_for_expert(expert_input)
                    
                    # user 메시지로 전문가 결과 추가
                    messages.append({
                        "role": "user",
                        "content": expert_result
                    })
                    
                    # 스트리밍으로 전문가 응답 표시
                    yield {"type": "tool_output", "content": expert_result, "input": expert_input}
                
                # <passage> 태그가 있으면 최종 응답으로 처리하고 종료
                if has_passage and remaining_text:
                    yield {"type": "final", "content": remaining_text}
                    break
                else:
                    # expert 호출 후 다음 라운드로 계속
                    continue
                    
            else:
                # expert 호출이 없는 경우
                if has_passage and remaining_text:
                    # <passage> 태그가 있으면 최종 응답으로 처리하고 종료
                    yield {"type": "final", "content": remaining_text}
                    break
                elif remaining_text:
                    # 일반 텍스트가 있으면 thinking으로 처리하고 계속
                    yield {"type": "think", "content": remaining_text}
                    continue
                else:
                    # 아무 텍스트가 없으면 다음 라운드로
                    continue
                
        except Exception as e:
            yield {"type": "think", "content": f"[error] {e}"}
            break

    # 최종 텍스트 결정
    if not final_text:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"]
                break
    
    # 혹시 빈 final response 방지
    if not final_text:
        yield {"type": "final", "content": final_text or ""}



# --- 타이핑 효과 함수 ---
async def typing_effect(text: str, placeholder, is_final: bool = False):
    """텍스트를 토큰 단위로 타이핑 효과를 내며 표시"""
    if not text:
        return
    
    # 단어 단위로 분할 (한국어와 영어 모두 고려)
    tokens = re.findall(r'\S+|\s+', text)
    
    displayed_text = ""
    for i, token in enumerate(tokens):
        displayed_text += token
        
        if is_final:
            # 최종 응답은 HTML 형식으로 표시
            final_response_content = f'<div class="final-response-container"><div class="passage-font-no-border">{format_text_to_html(displayed_text.strip())}</div></div>'
            placeholder.markdown(final_response_content, unsafe_allow_html=True)
        else:
            # 작가 모델 사고과정은 일반 마크다운으로 표시
            placeholder.markdown(displayed_text.strip() + " ▊")  # 커서 추가
        
        # 타이핑 속도 조절 (토큰 길이에 따라 조절)
        if len(token.strip()) > 0:  # 실제 내용이 있는 토큰만
            await asyncio.sleep(0.01)  # 30ms 딜레이
    
    # 최종적으로 커서 제거
    if not is_final:
        placeholder.markdown(displayed_text.strip())

# --- 스트리밍 실행 로직 ---
async def stream_and_render(final_user_prompt: str, selected_system_prompt: str):
    # 스트리밍 시작
    st.session_state["is_streaming"] = True
    
    try:
        # 시간순으로 모든 이벤트를 저장
        all_events = []
        final_content = ""
        
        # Vertex AI 설정값 사용
        endpoint_id = ENDPOINT_ID
        project_id = PROJECT_ID
        location = LOCATION
        
        # 동적으로 섹션을 추가할 메인 컨테이너
        with reasoning_placeholder.container():
            reasoning_main = st.container()
            
        expert_containers = {}  # 전문가 질의별 메인 컨테이너 저장
        
        async for event in run_vertex_ai_flow_streaming(
            endpoint_id=endpoint_id,
            project_id=project_id,
            location=location,
            system_prompt=selected_system_prompt,
            user_prompt=final_user_prompt,
            temperature=temperature,
        ):
            etype = event.get("type")
            
            if etype == "think":
                # 작가 모델 사고 과정 - 타이핑 효과 적용
                think_content = event.get("content", "").strip()
                if think_content:
                    with reasoning_main:
                        st.markdown("#### 작가 모델의 사고 과정")
                        thinking_placeholder = st.empty()
                    
                    # 타이핑 효과로 표시
                    await typing_effect(think_content, thinking_placeholder, is_final=False)
            
            elif etype == "tool_start":
                # 전문가 질의 시작 - 질의 내용만 먼저 표시
                input_text = (event.get("input") or "").strip()
                
                with reasoning_main:
                    expert_section = st.container()
                    with expert_section:
                        st.markdown("#### 전문가 모델에게 질의하기")
                        
                        # 질의 내용만 표시
                        with st.expander("질문 내용", expanded=False):
                            st.markdown(input_text)
                    
                    # 이 섹션을 저장해두어서 나중에 응답을 추가할 수 있도록 함
                    expert_containers[input_text] = expert_section
            
            elif etype == "tool_output":
                # 전문가 응답 완료 - 응답 익스팬더를 새로 생성
                out_text = (event.get("content") or "").strip()
                input_text = (event.get("input") or "").strip()
                
                # 해당 질의에 대한 컨테이너 찾아서 응답 추가
                if input_text in expert_containers:
                    with expert_containers[input_text]:
                        # 응답 내용 익스팬더를 새로 생성
                        with st.expander("응답 내용", expanded=False):
                            st.markdown(out_text)
            
            elif etype == "final":
                # 최종 응답 - 타이핑 효과 적용
                final_content = event.get("content", "").strip()
                if final_content:
                    await typing_effect(final_content, final_placeholder, is_final=True)
                break

    except Exception as e:
        st.error(f"모델 호출 또는 스트리밍 중 오류 발생: {e}")
    finally:
        # 스트리밍 종료
        st.session_state["is_streaming"] = False


# render_event 함수 제거됨 - 새로운 3컬럼 렌더링 방식 사용


# 실행 로직 - Preset 버튼 클릭 시
if 'preset_run_button' in locals() and preset_run_button:
    if 'system_prompt' in locals() and 'user_prompt' in locals() and system_prompt and user_prompt:
        final_user_prompt = user_prompt  # 원본 프롬프트 사용
        selected_system_prompt = system_prompt
        # 스트리밍 함수 실행
        asyncio.run(stream_and_render(final_user_prompt, selected_system_prompt))
    else:
        st.error("Preset 데이터를 로드할 수 없습니다.")

# 실행 로직 - Custom 버튼 클릭 시
if 'custom_run_button' in locals() and custom_run_button:
    custom_topic_content = st.session_state.get("custom_topic", "").strip()
    
    if custom_topic_content:
        # Custom 탭 데이터 사용
        custom_major_field_content = st.session_state.get("custom_major_field", "인문사회")
        custom_minor_field_content = st.session_state.get("custom_minor_field", "인문")
        custom_field_content = f"{custom_major_field_content} ({custom_minor_field_content})"
        custom_type_content = st.session_state.get("custom_type", "단일형")
        
        final_user_prompt = format_prompt_from_components(custom_field_content, custom_type_content, custom_topic_content)
        
        # 기본 시스템 프롬프트 사용 (첫 번째 샘플에서 추출)
        default_system_prompt, _, _ = load_sample(0)
        selected_system_prompt = default_system_prompt if default_system_prompt else ""
        
        # 스트리밍 함수 실행
        asyncio.run(stream_and_render(final_user_prompt, selected_system_prompt))
    else:
        st.error("주제를 입력해주세요.")
