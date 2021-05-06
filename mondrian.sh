while getopts d:m:s:n:t:y: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        m) mode=${OPTARG};;
        s) test=${OPTARG};;
        n) name=${OPTARG};;
        t) tag=${OPTARG};;
        y) tag2=${OPTARG};;
    esac
done

if [ "$mode" = "train" ]; then
    for i in {1..9}; do
        python3 mondrian_sequential.py --data $dataset --train_split 0."$i" -ws -t train -t 0."$i" -t $tag -t $tag2
    done
    python3 mondrian_sequential.py --data $dataset --train_split 1 -ws -t train -t 1 -t $tag -t $tag2
elif [ "$mode" = "test" ]; then
    for i in 1 2 3 4 5 6 7 8 9; do
        python3 evaluate_mondrian.py --data $dataset --train_split 0."$i" --test_size "$test" -ws -t test -t 0."$i" -t "$tag" -t "$tag2" -n "$i_$name_$test"
    done
    # python3 evaluate_mondrian.py --data $dataset --train_split 0."$i" --test_size $test -ws -t 0."$i" -t tag -t tag2
elif [ "$mode" = "bert" ]; then
    for i in {1..9}; do
        python3 bert_encoder.py --data $dataset --data_split 0."$i" -ws -t 0."$i" -t $tag -t $tag2
    done
    python3 bert_encoder.py --data $dataset --data_split 1 -ws -t 1 -t $tag -t $tag2
else
    echo "ERROR: use test or train mode"
fi