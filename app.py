"""♻️ 분리수거 비전 AI - Gradio demo.

    python app.py            # http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr

from src.config import ASSETS_DIR, CLASSES_KO
from src.inference import Predictor

GUIDE = {
    "cardboard": "**골판지/종이상자** - 테이프·운송장·철핀을 떼고 펼쳐서 납작하게 묶어 종이류로 배출하세요.",
    "glass": "**유리병** - 내용물을 비우고 헹군 뒤 뚜껑(플라스틱·금속)은 따로 분리해 유리병류로 배출하세요. "
             "깨진 유리·거울·도자기는 재활용이 안 되니 신문지에 싸서 종량제 봉투(또는 특수규격마대)에 버리세요.",
    "metal": "**캔/고철** - 내용물을 비우고 헹군 뒤 가능하면 압착해 캔류로 배출하세요. "
             "부탄가스·스프레이 캔은 가스를 완전히 뺀 뒤 배출하세요.",
    "paper": "**종이** - 물기에 젖지 않게 하고, 비닐 코팅·스프링·테이프를 제거한 뒤 묶어서 배출하세요. "
             "영수증·코팅지·오염된 종이는 일반쓰레기입니다.",
    "plastic": "**플라스틱** - 내용물을 비우고 헹군 뒤 라벨을 떼서 배출하세요. "
               "투명 페트병은 라벨을 제거하고 찌그러뜨려 별도 수거함에 배출하세요.",
    "trash": "**일반쓰레기** - 재활용이 어려운 품목입니다. 종량제 봉투에 담아 배출하세요.",
}
LOW_CONF = 0.5

predictor = Predictor()


def classify(image):
    if image is None:
        return None, None, "이미지를 올려 주세요."
    probs, cam = predictor.explain(image)
    labels = {f"{CLASSES_KO[c]} ({c})": p for c, p in probs.items()}
    top, top_p = max(probs.items(), key=lambda kv: kv[1])
    msg = f"### 예측: {CLASSES_KO[top]} ({top_p:.0%})\n\n{GUIDE[top]}"
    if top_p < LOW_CONF:
        msg += ("\n\n> ⚠️ 모델의 확신이 낮아요. 물체가 화면 중앙에 크게 보이도록, "
                "단색 배경에서 다시 찍어 보세요.")
    return labels, cam, msg


examples = sorted(str(p) for p in (ASSETS_DIR / "examples").glob("*.jpg"))

with gr.Blocks(title="분리수거 비전 AI") as demo:
    gr.Markdown(
        "# ♻️ 분리수거 비전 AI\n"
        "쓰레기 사진을 올리면 **6종(골판지·유리·금속·종이·플라스틱·일반쓰레기)** 중 무엇인지 분류하고, "
        "**Grad-CAM**으로 모델이 사진의 어디를 보고 판단했는지 보여줍니다.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="사진", sources=["upload", "webcam", "clipboard"])
            btn = gr.Button("분류하기", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="예측 확률")
            out_cam = gr.Image(label="Grad-CAM (빨간 영역일수록 판단에 크게 기여)")
            out_md = gr.Markdown()
    if examples:
        gr.Examples(examples, inputs=inp, outputs=[out_label, out_cam, out_md], fn=classify,
                    run_on_click=True, cache_examples=False, label="예시 이미지 (클릭하면 바로 분류)")
    btn.click(classify, inp, [out_label, out_cam, out_md])
    inp.upload(classify, inp, [out_label, out_cam, out_md])

if __name__ == "__main__":
    demo.launch()
