while getopts d:e:m: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        e) epochs=${OPTARG};;
        m) mode=${OPTARG};;
    esac
done

if [ "$mode" = "full" ] ; then
    python3 mondrian_sequential.py --data $dataset --epochs $epochs -ws --full_train
fi

python3 mondrian_sequential.py --data $dataset --epochs $epochs -ws 