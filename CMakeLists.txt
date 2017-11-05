cmake_minimum_required(VERSION 2.8.9)
project(userOp)

#-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -march=native -shared ten_add.cc -o ten_add.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2

add_definitions(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -march=native -fPIC -O2)



include_directories(/usr/local/lib/python3.5/dist-packages/tensorflow/include)
include_directories(/usr/local/lib/python3.5/dist-packages/tensorflow/include/external/nsync/public)

link_directories(/usr/local/lib/python3.5/dist-packages/tensorflow)
link_directories(tensorflow_framework)

add_library(cmake_ten_add SHARED ten_add.cc)