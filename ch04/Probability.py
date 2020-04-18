def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:
    #균등 분포를 따르는 확률변수의 값이 x보다 작거나 같은 확률을 반환
    if x < 0: return 0 # 균등 분포의 확률은 절대로 0보다 작을 수 없다.
    elif x < 1: return x
    else: return 1 # 균등 분포의 확률은 항상 1보다 작다.


#실습
def updf(x: float) -> float:
    return 3 if 2 <= x < 3 else 2

def ucdf(x: float) -> float:
    #균등 분포를 따르는 확률변수의 값이 x보다 작거나 같은 확률을 반환
    if x < 2: return 2
    elif x < 3: return x
    else: return 3

