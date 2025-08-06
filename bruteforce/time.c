#include <stdio.h>
#include <time.h>

#define N 26  // Jumlah huruf alfabet

int main() {
    char letters[N];
    for (int i = 0; i < N; i++) {
        letters[i] = 'A' + i;
    }

    clock_t start = clock();

    long long count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    for (int m = 0; m < N; m++) {
                        for (int n = 0; n < N; n++) {
                            for (int o = 0; o < N; o++) {
                                for (int p = 0; p < N; p++) {
                                    printf("%c%c%c%c%c%c%c%c\n", letters[i], letters[j], letters[k], letters[l], letters[m], letters[n], letters[o], letters[p]);
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTotal kombinasi: %lld\n", count);
    printf("Waktu eksekusi: %.6f detik\n", time_taken);
    printf("Kompleksitas waktu: O(n^8), dengan n = %d\n", N);

    return 0;
}
