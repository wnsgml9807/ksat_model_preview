import streamlit as st
import asyncio
import json
from openai import AsyncOpenAI, OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from transformers import AutoTokenizer

# --- 페이지 기본 설정 ---
st.set_page_config(layout="wide", page_title="모델 지문 생성 뷰어")

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
    font-family: 'Nanum Myeongjo', serif !important;
    line-height: 1.7;
    letter-spacing: -0.01em;
    font-weight: 500;
}
.passage-font p {
    text-indent: 1em;
    margin-bottom: 0em;
    margin-top: 0.5em;
}
</style>
""", unsafe_allow_html=True)


# --- 설정값 ---
DATASET_PATH = "GPT-sft-09-03-val.jsonl"
# OpenAI fine-tuned model 기본값 (필요 시 사이드바에서 변경)
MODEL_NAME = "ft:gpt-4.1-2025-04-14:ksat-agent:exp-09-03-large:CBdqVMpr"


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
            
            system_prompt = data["messages"][0]["content"]
            user_prompt = data["messages"][1]["content"]
            expected_response_raw = data["messages"][-1]["content"]
            
            if "<think>" in expected_response_raw:
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


# --- 사이드바 UI ---
st.sidebar.title("⚙️ 모델 및 데이터 설정")

total_samples = get_dataset_info(DATASET_PATH)
if total_samples > 0:
    dataset_index = st.sidebar.selectbox(
        "데이터 샘플 선택",
        options=range(total_samples),
        format_func=lambda x: f"Sample #{x}",
        index=0
    )
else:
    st.sidebar.error("데이터셋 파일을 찾을 수 없습니다. `DATASET_PATH`를 확인하세요.")
    st.stop()

model_name = st.sidebar.text_input("모델 이름 (OpenAI)", value=MODEL_NAME)
tokenizer_path = st.sidebar.text_input("토크나이저 경로 (옵션)", value=model_name)


st.sidebar.divider()
st.sidebar.subheader("생성 파라미터")
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95, 0.05)
presence_penalty = st.sidebar.slider("Presence Penalty", -1.0, 1.0, 0.0, 0.05)

run_button = st.sidebar.button("🚀 지문 생성 시작", type="primary")


# --- 메인 패널 UI ---
col1, col2 = st.columns(2)
system_prompt, user_prompt, expected_response = load_sample(dataset_index)

with col1:
    st.subheader("원본 지문")
    with st.container(border=True, height=600):
        if expected_response:
            html_expected = f'<div class="passage-font">{format_text_to_html(expected_response)}</div>'
            st.markdown(html_expected, unsafe_allow_html=True)
        else:
            st.warning("선택된 인덱스의 데이터를 불러올 수 없습니다.")

with col2:
    st.subheader("모델 응답")
    # 스트림릿 컨테이너에 보더 적용
    with st.container(border=True, height=600):
        placeholder = st.empty()
        # 초기 안내 메시지
        with placeholder.container():
            st.markdown(f'<div class="passage-font">왼쪽 설정 패널에서 "지문 생성 시작" 버튼을 눌러주세요.</div>', unsafe_allow_html=True)


# --- 도구 함수 (전문가 호출) ---
async def execute_request_for_expert(input_text: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "[expert_error] Missing GOOGLE_API_KEY"

    def _call_sync() -> str:
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            resp = client.chat.completions.create(
                model="gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": input_text}],
            )
            return resp.choices[0].message.content if resp.choices else ""
        except Exception as e:
            return f"[expert_error] {e}"

    result = await asyncio.to_thread(_call_sync)
    return result or "[expert_empty]"


# --- think/call 루프 (스트리밍 이벤트 생성) ---
async def run_tool_call_flow_streaming(model_name: str, system_prompt: str, user_prompt: str, temperature: float, top_p: float, presence_penalty: float):
    client = AsyncOpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "request_for_expert",
                "description": "Request information from an expert assistant and return textual content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Instruction or query text for the expert LLM.",
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        }
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    round_idx = 0
    turn_type = "think"
    final_text = ""

    while True:
        round_idx += 1
        if round_idx > 30:
            break

        if turn_type == "think":
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    tools=tools,
                    tool_choice="none",
                )
                content = response.choices[0].message.content or ""
                if content:
                    # 최종 응답 판단: </think> 이후 텍스트가 있고 도구 호출이 아닌 경우
                    if "</think>" in content:
                        before, after = content.split("</think>", 1)
                        reasoning_text = before
                        if "<think>" in reasoning_text:
                            reasoning_text = reasoning_text.split("<think>", 1)[1]
                        if reasoning_text.strip():
                            yield {"type": "think", "content": reasoning_text.strip()}
                        
                        after_text = after.strip()
                        # 도구 호출이 아닌 실제 텍스트가 있는지 확인
                        if after_text and not after_text.startswith("<tool"):
                            # 중요: assistant 메시지를 먼저 추가한 후 최종 응답 처리
                            messages.append({"role": "assistant", "content": content})
                            yield {"type": "final", "content": after_text}
                            break
                        else:
                            # </think> 이후 도구 호출이 있는 경우
                            messages.append({"role": "assistant", "content": content})
                            turn_type = "call"
                            continue
                    else:
                        yield {"type": "think", "content": content}
                        messages.append({"role": "assistant", "content": content})
                        turn_type = "call"
                        continue
                turn_type = "call"
            except Exception as e:
                yield {"type": "think", "content": f"[error - think] {e}"}
                break

        else:  # call turn
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "request_for_expert"}},
                )

                assistant_message = response.choices[0].message
                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls or []

                # 간헐적 텍스트는 reasoning에 합류하거나 최종 응답 판단 (도구 호출 처리 후 종료)
                has_final_response = False
                final_response_text = ""
                
                if content:
                    if "</think>" in content:
                        before, after = content.split("</think>", 1)
                        reasoning_text = before
                        if "<think>" in reasoning_text:
                            reasoning_text = reasoning_text.split("<think>", 1)[1]
                        if reasoning_text.strip():
                            yield {"type": "think", "content": reasoning_text.strip()}
                        
                        after_text = after.strip()
                        # 도구 호출이 아닌 실제 텍스트가 있는지 확인 (바로 break하지 않음)
                        if after_text and not after_text.startswith("<tool"):
                            has_final_response = True
                            final_response_text = after_text
                    else:
                        yield {"type": "think", "content": content}

                msg_to_add = {"role": "assistant", "content": content}
                if tool_calls:
                    msg_to_add["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in tool_calls
                    ]
                messages.append(msg_to_add)

                if tool_calls:
                    for tool_call in tool_calls:
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except Exception:
                            args = {"input": tool_call.function.arguments}
                        # 도구 호출 질문도 함께 전달
                        tool_input = args.get("input", "")
                        tool_output_content = await execute_request_for_expert(tool_input)
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": tool_output_content,
                        })

                        yield {"type": "tool_output", "content": tool_output_content, "input": tool_input}

                    # 도구 호출 처리 후 최종 응답이 있다면 출력하고 종료
                    if has_final_response:
                        yield {"type": "final", "content": final_response_text}
                        break
                    else:
                        turn_type = "think"  # 다음 사고 턴으로
                else:
                    # 도구 호출이 없는 경우
                    if has_final_response:
                        yield {"type": "final", "content": final_response_text}
                        break
                    elif content:
                        # 최종 응답도 아니고 도구 호출도 없으면 종료
                        yield {"type": "final", "content": content}
                        break
                    else:
                        break
            except Exception as e:
                yield {"type": "think", "content": f"[error - call] {e}"}
                break

    # 최종 텍스트 결정
    if not final_text:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"]
                break
    yield {"type": "final", "content": final_text or ""}


# --- 스트리밍 실행 로직 ---
async def stream_and_render():
    try:
        # 토크나이저는 선택 사항 (길이 계산 실패 시 안전하게 무시)
        try:
            _ = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            pass

        # 각 이벤트마다 새로운 컨테이너를 생성해서 복제 문제 방지
        event_containers = []
        
        async for event in run_tool_call_flow_streaming(
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
        ):
            etype = event.get("type")
            
            # 기존 placeholder를 비우고 새로운 컨테이너 생성
            with placeholder.container():
                # 이전 이벤트들을 다시 렌더링
                for prev_event in event_containers:
                    render_event(prev_event)
                
                # 현재 이벤트 렌더링
                render_event(event)
                
                # 이벤트 저장
                event_containers.append(event)
                
                if etype == "final":
                    break

    except Exception as e:
        st.error(f"모델 호출 또는 스트리밍 중 오류 발생: {e}")


def render_event(event):
    """개별 이벤트를 렌더링하는 함수"""
    etype = event.get("type")
    
    if etype == "think":
        st.markdown("### 🤔 Reasoning:")
        st.markdown(f'<div class="passage-font">{format_text_to_html((event.get("content") or "").strip())}</div>', unsafe_allow_html=True)
        
    elif etype == "tool_output":
        out_text = (event.get("content") or "").strip()
        input_text = (event.get("input") or "").strip()
        st.markdown("### 🔍 Request for Expert")
        with st.expander("📤 Question to Expert", expanded=False):
            st.markdown(input_text)
        with st.expander("🧠 Expert's response", expanded=False):
            st.markdown(out_text)

    elif etype == "final":
        st.markdown("### ✅ Final Response:")
        st.markdown(f'<div class="passage-font">{format_text_to_html((event.get("content") or "").strip())}</div>', unsafe_allow_html=True)


if run_button:
    # 버튼이 클릭되면 스트리밍 함수를 실행
    asyncio.run(stream_and_render())
