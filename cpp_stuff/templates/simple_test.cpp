#include <iostream>

template <typename T>
T max(T const a , T const b){
    return a > b ? 1 : 2;
}


int main(){


std::cout<<max<int>(1,1);
std::cout<<max<double>(3,2);
return 0;
}