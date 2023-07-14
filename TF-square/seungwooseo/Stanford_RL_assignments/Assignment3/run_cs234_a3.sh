declare -a env=("cheetah")
declare -a seeds=("42" "43" "44")

## now loop through the above array
for i in "${env[@]}"
do
	for j in "${seeds[@]}"
	do
		python main.py --env-name "$i" \
			--seed "$j" \
			--no-baseline

		python main.py --env-name "$i" \
                        --seed "$j" \
                        --baseline
	done
	python plot.py --env-name "$i" \
		--seeds "42,43,44"
done
