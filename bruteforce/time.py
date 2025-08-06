import time
import string

N = 26
letters = list(string.ascii_uppercase)

start = time.time()

count = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for m in range(N):
                    for n in range(N):
                        for o in range(N):
                            for p in range(N):
                                # print(f"{letters[i]}{letters[j]}{letters[k]}{letters[l]}{letters[m]}{letters[n]}{letters[o]}{letters[p]}")
                                count += 1

end = time.time()
time_taken = end - start

print(f"\nTotal kombinasi: {count}")
print(f"Waktu eksekusi: {time_taken:.6f} detik")
print(f"Kompleksitas waktu: O(n^8), dengan n = {N}")
