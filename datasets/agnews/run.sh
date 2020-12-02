for keyword in politics sports business technology
do
        for number_of_loop in 1 2 3
        do
                mkdir -p $keyword$number_of_loop
                echo $keyword > label_name_loop.txt
                python3 run.py --keyword $keyword --number_of_loop $number_of_loop
        done
done

