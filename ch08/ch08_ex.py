# 알파벳, 숫자, 어프스트로피(')가 포함된 단어를 추출

from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower();
    all_words = re.findall("[a-z0-9']+", text)
    return set(all_words)

print(tokenize("Data Science is science"))


# 학습 데이터 저장을 위한 클래스 정의
from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool


from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set() # 메시지에 나타나는 모든 토큰들 저장
        self.token_spam_counts: Dict[str, int] = defaultdict(int)#토큰이 스팸에 나타난 빈도
        self.token_ham_counts: Dict[str, int] = defaultdict(int) #토큰이 햄에 나타난 빈도
        self.spam_messages = self.ham_messages = 0  # 각 메시지종류별 메시지 수

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    # P(token|스팸), P(token|햄) 확률 리턴
    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
        return p_token_spam, p_token_ham

    # log, exp를 활용한 P(스팸|tokens) 예측
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:  # 훈련에서 얻은 모든 토큰에 대해서 루프 처리
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it;
            if token in text_tokens:  # 훈련에서 얻은 토큰이 검토중인 문자열에 나타나면
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:  # 훈련에서 얻은 토큰이 검토중인 문자열에 없으면
                # (훈련 데이터 토큰을 항상 모두 쓰기위한 조치임), # P3의 (1)식 유도참고
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)  # P3의 (1)식 유도


# 학습데이터 만들기
messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]
model = NaiveBayesClassifier(k=0.5)
model.train(messages)
# 훈련데이터 모델 확인 : assert  print로 교체해서 테스트
assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}
# 메시지 “hello spam”의 스팸여부 판단 예 : 수작업으로 구하고, 구현 값과 비교
text = "hello spam"
probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),  # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)  # "hello" (present)
]
probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),  # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
]
p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# print로 교체해서 비교: Should be about 0.83
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham)

print(model.predict(text))
print(p_if_spam / (p_if_spam + p_if_ham))


# 복수형 단어를 단수형으로 통일
def drop_final_s(word):
    return re.sub("s$", "", word)


def main():
    import glob, re

    # modify the path to wherever you've put the files
    path = 'spam_data/*/*'

    data: List[Message] = []

    # glob.glob returns every filename that matches the wildcarded path
    for filename in glob.glob(path):
        is_spam = "ham" not in filename

        # There are some garbage characters in the emails, the errors='ignore'
        # skips them instead of raising an exception.
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  # done with this file

    import random
    from scratch.machine_learning import split_data

    random.seed(0)      # just so you get the same answers as me
    train_messages, test_messages = split_data(data, 0.75)

    model = NaiveBayesClassifier()
    model.train(train_messages)

    from collections import Counter

    predictions = [(message, model.predict(message.text))
                   for message in test_messages]

    # Assume that spam_probability > 0.5 corresponds to spam prediction
    # and count the combinations of (actual is_spam, predicted is_spam)
    confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                               for message, spam_probability in predictions)

    print(confusion_matrix)

    # 주어진 모델에서 P(스팸|token) 확률 계산
    def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
        # We probably shouldn＇t call private methods, but it＇s for a good cause.
        prob_if_spam, prob_if_ham = model._probabilities(token)

        return prob_if_spam / (prob_if_spam + prob_if_ham)

    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])

if __name__ == "__main__": main()