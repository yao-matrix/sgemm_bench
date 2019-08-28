wget https://github.com/intel/mkl-dnn/releases/download/v0.20/mklml_lnx_2019.0.5.20190502.tgz
tar -xzvf ./mklml_lnx_2019.0.5.20190502.tgz
ln -s ./mklml_lnx_2019.0.5.20190502 ./mklml


cd ./src
make



