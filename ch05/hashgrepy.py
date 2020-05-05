# 첫 글자가 #으로 시작하는 라인수 카운트
import re

starts_with_hash = 0;
with open('input.txt', 'r') as f:
    for line in f:
        if re.match("^#", line):
            starts_with_hash += 1
print(starts_with_hash)