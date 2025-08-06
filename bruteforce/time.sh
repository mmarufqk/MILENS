#!/bin/bash

letters=( {A..Z} )

start_time=$(date +%s)

count=0

for i in "${letters[@]}"; do
    for j in "${letters[@]}"; do
        for k in "${letters[@]}"; do
            for l in "${letters[@]}"; do
                for m in "${letters[@]}"; do
                    for n in "${letters[@]}"; do
                        for o in "${letters[@]}"; do
                            for p in "${letters[@]}"; do
                                echo "$i$j$k$l$m$n$o$p"
                                count=$((count + 1))
                            done
                        done
                    done
                done
            done
        done
    done
done

end_time=$(date +%s)
time_taken=$((end_time - start_time))

echo ""
echo "Total kombinasi: $count"
echo "Waktu eksekusi: $time_taken detik"
echo "Kompleksitas waktu: O(n^8), dengan n = ${#letters[@]}"
