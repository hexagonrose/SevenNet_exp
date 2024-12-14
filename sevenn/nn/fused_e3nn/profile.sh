#!/bin/bash
#!/bin/bash


L_MAX=(1 2 3)
H=(32)
Layer=(0 1 2 3)
Batch=(8192 16384 32768 65536 131072)

py_name=$1
version=$2

dirname="./"
mkdir -p ${dirname}/txt
mkdir -p ${dirname}/nsys
mkdir -p ${dirname}/sqlite
mkdir -p ${dirname}/processed

## nsys eval
for lmax in ${L_MAX[@]}
do
	for h in ${H[@]}
    do
        for layer_idx in ${Layer[@]}
        do
            for b in ${Batch[@]}
            do
                filename="${version}_${h}_${lmax}_${layer_idx}_${b}"            
                if [ ! -e "${dirname}/nsys/${filename}.nsys-rep" ]; then
                echo $filename
                CUDA_VISIBLE_DEVICES=1 nsys profile -f true --capture-range=cudaProfilerApi -o ${dirname}/nsys/${filename} python ${py_name} ${h} ${lmax} ${layer_idx} tp ${b} False &> ${dirname}/txt/${filename}.txt
                fi
            done
        done
	done
done


## sqlite extract
for lmax in ${L_MAX[@]}
do
	for h in ${H[@]}
    do
        for layer_idx in ${Layer[@]}
        do
            for b in ${Batch[@]}
            do
                filename="${version}_${h}_${lmax}_${layer_idx}_${b}"            
                if [ ! -e "${dirname}/sqlite/${filename}.sqlite" ]; then
                nsys export -f true -t sqlite -o ${dirname}/sqlite/${filename}.sqlite ${dirname}/nsys/${filename}.nsys-rep
                fi
            done
        done
	done
done

