import hashlib, sys

TARGET_HASH = "6998506d94a687fcac1d80ee846763ba" 
h = TARGET_HASH.strip().lower()

for i in range(1000000):
    pwd = i
    if hashlib.md5(str(pwd).encode()).hexdigest() == h:
        print(i)
        sys.exit(0)

