"""
Create slide-ready overview images for the manuscript figures.

Outputs:
- figures/kci/kci_figure_caption_overview.png
- figures/kci/kci_figure_thumbnail_panel.png
"""
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
FIGDIR = BASE / "figures"
OUTDIR = FIGDIR / "kci"
OUTDIR.mkdir(parents=True, exist_ok=True)


FIGURES = [
    {
        "no": "그림 1",
        "title": "초기 관측 길이 K에 따른 딥러닝 모델 성능 변화",
        "desc": "K가 5에서 14로 증가함에 따라 LSTM, GRU, Transformer, TFT 모두 ROC-AUC가 상승하는 경향을 보인다.",
        "path": OUTDIR / "kci_fig2_k_ablation.png",
    },
    {
        "no": "그림 2",
        "title": "주요 모델별 ROC-AUC 비교",
        "desc": "CatBoost-Full (K=14)가 가장 높은 ROC-AUC를 기록했으며, XGBoost와 LightGBM도 유사한 성능을 보인다.",
        "path": OUTDIR / "kci_fig1_model_auc_ci.png",
    },
    {
        "no": "그림 3",
        "title": "paired bootstrap 기반 모델 간 AUC 차이",
        "desc": "CatBoost-Full (K=14)와 Transformer (K=14)의 AUC 차이는 유의하나, CatBoost-Full과 XGBoost-Full의 차이는 유의하지 않다.",
        "path": OUTDIR / "kci_fig3_bootstrap_diff.png",
    },
    {
        "no": "그림 4",
        "title": "CatBoost-Full (K=14) 특징 중요도",
        "desc": "카테고리, 초기 구간의 순위 수준, 태그 수, 조회 속도 변화 관련 특징이 상위에 위치한다.",
        "path": OUTDIR / "kci_fig4_feature_importance.png",
    },
    {
        "no": "그림 5",
        "title": "TFT attention 시각화",
        "desc": "딥러닝 모델이 초기 관측 구간 내 특정 시점 간 관계를 활용하는 양상을 보여준다.",
        "path": FIGDIR / "fig5_tft_attention.png",
    },
    {
        "no": "그림 6",
        "title": "주요 모델의 Precision-Recall curve",
        "desc": "클래스 불균형 조건에서 모델별 양성 클래스 탐지 성능을 비교한다.",
        "path": FIGDIR / "fig6_pr_curve.png",
    },
    {
        "no": "그림 7",
        "title": "기준선 및 특징군 제거 실험",
        "desc": "첫 스냅샷만 사용한 모델보다 초기 14개 관측의 시계열 요약 특징을 사용한 모델의 성능이 높다.",
        "path": OUTDIR / "kci_fig5_ablation_baselines.png",
    },
    {
        "no": "그림 8",
        "title": "바이럴 임계값 민감도 실험",
        "desc": "p65~p85 기준에서 ROC-AUC가 안정적으로 유지되어, 결과가 특정 p75 기준에만 의존하지 않음을 보여준다.",
        "path": OUTDIR / "kci_fig6_threshold_sensitivity.png",
    },
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothicBold.ttf" if bold else "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = font(44, True)
FONT_SUB = font(22)
FONT_NO = font(24, True)
FONT_CARD_TITLE = font(20, True)
FONT_DESC = font(17)
FONT_SMALL = font(14)


def rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(draw, text, xy, max_width, fnt, fill, line_gap=6):
    x, y = xy
    lines = []
    for para in text.split("\n"):
        current = ""
        for char in para:
            test = current + char
            bbox = draw.textbbox((0, 0), test, font=fnt)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def make_caption_overview():
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), "#F7F9FC")
    draw = ImageDraw.Draw(img)

    draw.text((80, 54), "논문 그림 구성 및 해석 요약", font=FONT_TITLE, fill="#0B2545")
    draw.text(
        (82, 112),
        "초기 트렌딩 스냅샷 기반 YouTube 트렌딩 지속성 조기 예측 연구",
        font=FONT_SUB,
        fill="#4A5568",
    )

    cols, rows = 2, 4
    margin_x, margin_top = 80, 175
    gap_x, gap_y = 34, 24
    card_w = (W - margin_x * 2 - gap_x) // cols
    card_h = (H - margin_top - 70 - gap_y * (rows - 1)) // rows

    for idx, item in enumerate(FIGURES):
        c, r = idx % cols, idx // cols
        x = margin_x + c * (card_w + gap_x)
        y = margin_top + r * (card_h + gap_y)
        rounded_rect(draw, (x, y, x + card_w, y + card_h), 18, "#FFFFFF", "#D7DEE8", 2)
        rounded_rect(draw, (x + 20, y + 22, x + 116, y + 60), 12, "#E8F1FF", None)
        draw.text((x + 38, y + 28), item["no"], font=FONT_NO, fill="#1F4D78")
        draw.text((x + 135, y + 26), item["title"], font=FONT_CARD_TITLE, fill="#102033")
        draw_wrapped(draw, item["desc"], (x + 28, y + 82), card_w - 56, FONT_DESC, "#2D3748", 7)

    img.save(OUTDIR / "kci_figure_caption_overview.png", quality=95)
    print(OUTDIR / "kci_figure_caption_overview.png")


def fit_image(path, size):
    W, H = size
    if not path.exists():
        return Image.new("RGB", size, "#EEF2F7")
    im = Image.open(path).convert("RGB")
    im.thumbnail((W, H), Image.LANCZOS)
    canvas = Image.new("RGB", size, "#FFFFFF")
    x = (W - im.width) // 2
    y = (H - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def make_thumbnail_panel():
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 130), fill="#0B2545")
    draw.text((70, 38), "논문 사용 그림 패널", font=FONT_TITLE, fill="#FFFFFF")
    draw.text((72, 92), "8개 핵심 그림과 해석 캡션", font=FONT_SUB, fill="#DCE8F7")

    cols, rows = 4, 2
    margin_x, margin_top = 50, 160
    gap_x, gap_y = 24, 26
    card_w = (W - margin_x * 2 - gap_x * (cols - 1)) // cols
    card_h = (H - margin_top - 44 - gap_y * (rows - 1)) // rows
    thumb_h = 250

    for idx, item in enumerate(FIGURES):
        c, r = idx % cols, idx // cols
        x = margin_x + c * (card_w + gap_x)
        y = margin_top + r * (card_h + gap_y)
        rounded_rect(draw, (x, y, x + card_w, y + card_h), 14, "#F8FAFD", "#D7DEE8", 2)
        thumb = fit_image(item["path"], (card_w - 24, thumb_h))
        img.paste(thumb, (x + 12, y + 12))
        draw.text((x + 18, y + thumb_h + 24), f'{item["no"]}. {item["title"]}', font=FONT_CARD_TITLE, fill="#0B2545")
        draw_wrapped(draw, item["desc"], (x + 18, y + thumb_h + 58), card_w - 36, FONT_SMALL, "#2D3748", 4)

    img.save(OUTDIR / "kci_figure_thumbnail_panel.png", quality=95)
    print(OUTDIR / "kci_figure_thumbnail_panel.png")


if __name__ == "__main__":
    make_caption_overview()
    make_thumbnail_panel()
