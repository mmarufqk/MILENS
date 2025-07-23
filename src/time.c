#include <stdio.h>
#include <time.h>

#define N 26

int main() {
    char letters[N];
    for (int i = 0; i < N; i++) {
        letters[i] = 'A' + i;
    }

    clock_t start = clock();

    int count = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                for (int l = 0; l < N; l++) {
                    for (int m = 0; m < N; m++) {
                        printf("%c%c%c%c%c\n", letters[i], letters[j], letters[k], letters[l], letters[m]);
                        count++;
                    }
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTotal kombinasi: %d\n", count);
    printf("Waktu eksekusi: %.6f detik\n", time_taken);
    printf("Kompleksitas waktu: O(n^5), dengan n = %d\n", N);

    return 0;
}
