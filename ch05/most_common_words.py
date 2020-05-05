# most_common_words.py
import sys
from collections import Counter

if __name__ == "__main__":

    # pass in number of words as first argument
    try:
        num_words = int(sys.argv[1])
    except:
        print("usage: most_common_words.py num_words")
        sys.exit(1)   # non-zero exit code indicates error

    counter = Counter(word.lower()
                      for line in sys.stdin
                      for word in line.strip().split()
                      if word)

    for word, count in counter.most_common(num_words): # Counter.most_common은 데이터의 개수가 많은 순으로 정렬된 배열을 리턴
        sys.stdout.write(str(count))
        sys.stdout.write("\t")
        sys.stdout.write(word)
        sys.stdout.write("\n")
