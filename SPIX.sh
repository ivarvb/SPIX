#!/bin/bash

ev="./python/python38spix/"
install () {
    ###requery

    #sudo apt-get install python3-venv
    #sudo apt install python3-pip
    #sudo apt-get install libsuitesparse-dev
    #sudo apt install libx11-dev
    #############sudo apt install nvidia-cuda-toolkit


    rm -r $ev
    mkdir $ev
    python3 -m venv $ev
    source $ev"bin/activate"

    # install packages
    # pip3 install -r requirements.txt
    pip3 install wheel
    pip3 install numpy
    #pip3 install scikit-sparse
    pip3 install matplotlib
    pip3 install pandas
    #pip3 install opencv-python
    #pip3 install scikit-image
    #pip3 install -U scikit-learn
    #pip3 install xgboost
    pip3 install ujson
    #pip3 install seaborn
    pip3 install cython
    #pip3 install xgboost

    pip3 install SimpleITK
   # pip3 install pyradiomics
    #pip3 install thundersvm-cpu
    
    compile
}
compile () {
    source $ev"bin/activate"

    #cd ./sourcecode/src/vx/com/px/descriptor/bic/
    #sh Makefile.sh
    #cd ../../../../../../../

    cd ./sourcecode/src/vx/com/px/image/
    sh Makefile.sh

    #cd ../../../../../../
    #cd ./sourcecode/src/vx/com/px/superpixels/snic/
    #sh Makefile.sh

}
execute () {
    source $ev"bin/activate"

    cd ./sourcecode/src/
    python3 SPIX.py
}

args=("$@")
T1=${args[0]}
if [ "$T1" = "install" ]; then
    install
elif [ "$T1" = "compile" ]; then
    compile
elif [ "$T1" = "execute" ]; then
    execute
fi






