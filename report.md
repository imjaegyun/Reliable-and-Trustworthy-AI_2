# 공격 기법과 신경망 테스트의 연결

FGSM, PGD와 DeepXplore는 모두 모델의 약점을 찾기 위해 기울기를 활용한다는
공통점이 있다. 이번 실험에서도 CIFAR-10 입력 중 일부는 원본 이미지 상태에서
두 ResNet50 모델의 예측이 달랐고, 일부는 작은 perturbation을 준 뒤에 예측이
갈라졌다. 과제에서는 threshold 값을 따로 지정하지 않았기 때문에 0.2, 0.5,
0.75, 0.9를 함께 비교했다. 같은 30개 seed와 20회 탐색 조건에서 네 threshold
모두 30개의 예측 불일치 입력을 찾았다. 즉 이번 두 모델 조합에서는 threshold를
바꿔도 disagreement 발견 수는 줄지 않았다. 다만 평균 coverage는 1.000,
0.966, 0.345, 0.089로 달라졌고, 0.2와 0.5에서는 coverage가 거의 포화된 반면
0.9는 지나치게 엄격했다. 최종 실행에서는 0.75를 대표 threshold로 두었다. 이
점에서 adversarial attack은 differential testing의 좋은 seed를
만드는 방법이 될 수 있다. 깨끗한 테스트 이미지에서만 시작하는 대신, FGSM이나
PGD로 이미 decision boundary 근처에 있는 입력을 만든 뒤 DeepXplore의 탐색
초기값으로 사용할 수 있다.

반대로 DeepXplore의 neuron coverage 개념을 공격에 넣는 것도 가능하다. 일반적인
FGSM이나 PGD는 주로 misclassification loss를 키우는 방향으로 입력을 수정한다.
하지만 여기에 아직 활성화되지 않은 뉴런을 활성화하는 항을 추가하면,
misclassification과 coverage를 동시에 높이는 coverage-guided attack을 만들 수
있다. 이렇게 하면 공격은 단순히 하나의 취약한 방향만 반복해서 찾는 것이 아니라,
네트워크 내부의 다양한 동작 영역을 탐색하게 된다.

두 방법을 결합하면 효율성도 좋아질 수 있다. PGD는 입력을 빠르게 failure
region으로 이동시키는 데 강하고, neuron coverage는 탐색이 한정된 영역에만
머무르는 것을 줄여준다. 여러 모델을 동시에 테스트할 때는 한 모델의 원래 label
confidence를 낮추면서, 두 모델의 coverage를 함께 높이는 목적함수를 사용할 수
있다. 그 결과 모델들의 예측이 달라지면 suspicious test case로 기록할 수 있다.

물론 항상 도움이 되는 것은 아니다. Coverage metric이 너무 단순하면 실제 의미
있는 오류가 아니라 단순히 activation만 다른 입력을 찾을 수 있다. 또한
perturbation이 지나치게 커지면 사람이 보기에는 비현실적인 이미지가 되어
실제 신뢰성 평가와 거리가 생긴다. 따라서 두 방법을 결합할 때는 공격 기울기로
탐색 속도를 높이되, perturbation 크기를 제한하고 생성된 이미지를 직접
확인하는 절차가 필요하다.
