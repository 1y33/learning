#include <iostream>

// function tempalte
template <typename T>
T add(T const a, T const b){
    return a + b ;
}


template <typename Input, typename Predicate>
int count_if(Input start, Input end , Predicate p){
    int total = 0;
    for (Input i = start ;  i!= end ; i ++){
        if (p(*i))
            total++;
    }
    return total;
}
int main(){

    auto d = add<double>(41.2,1);
    std::cout<<d;

    ///
    int arr[]{1,2,3,4,4,4,4,4,4};
    int odds = count_if(
        std::begin(arr),std::end(arr),[](int const n){
            return n%2 == 1;
        }
    );
    std::cout<<"\nODDS "<<odds;
    return 0;
}