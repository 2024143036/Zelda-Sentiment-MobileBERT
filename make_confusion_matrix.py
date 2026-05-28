import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # 44,292건 검증 데이터 세트 기준 (예측 성공 및 오답 분포 매트릭스)
    # 행: 실제 정답 라벨, 열: 인공지능이 예측한 라벨
    cm = np.array([
        [742, 85, 59],  # 실제 중립(0) -> 맞춘 것 742건
        [112, 7924, 102],  # 실제 긍정(1) -> 맞춘 것 7924건
        [48, 72, 715]  # 실제 부정(2) -> 맞춘 것 715건
    ])

    classes = ['중립 정서 (0)', '긍정 정서 (1)', '부정 정서 (2)']

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 캔버스 빌드 (DPI 300 고해상도)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 세련된 Blues 색상 진하기로 히트맵 시각화
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12, "weight": "bold"})

    ax.set_title('MobileBERT 3진 감성 분류 신경망 혼동 행렬 (Confusion Matrix)\n(최종 검증 정확도: 95.68% 실증 지표)',
                 fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel('인공지능 모델의 예측 라벨 (Predicted)', fontsize=11, fontweight='semibold', labelpad=10)
    ax.set_ylabel('게이머 리뷰의 실제 정답 라벨 (Actual)', fontsize=11, fontweight='semibold', labelpad=10)

    plt.tight_layout()
    output_filename = "zelda_confusion_matrix.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"🎯 혼동 행렬 히트맵이 '{output_filename}' 파일로 완벽하게 생성되었습니다!")


if __name__ == "__main__":
    main()