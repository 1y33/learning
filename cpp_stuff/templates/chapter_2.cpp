#include <iostream>

// function tempalte
template <typename T>
T add(T const a, T const b)
{
    return a + b;
}

template <typename Input, typename Predicate>
int count_if(Input start, Input end, Predicate p)
{
    int total = 0;
    for (Input i = start; i != end; i++)
    {
        if (p(*i))
            total++;
    }
    return total;
}

/// class template
//
template <typename T>
class wrapper
{
    T value;

public:
    wrapper(T val) : value(val) {}
};

void use_wrapper(wrapper<int> *ptr) {}

/// templated classes and also templated members

template <typename T>
class number
{
    T value;

public:
    number(T val) : value(val) {}

    template <typename U>
    U as()
    {
        return static_cast<U>(value);
    }
};

int main()
{

    auto d = add<double>(41.2, 1);
    std::cout << d;

    ///
    int arr[]{1, 2, 3, 4, 4, 4, 4, 4, 4};
    int odds = count_if(
        std::begin(arr), std::end(arr), [](int const n)
        { return n % 2 == 1; });
    std::cout << "\nODDS " << odds;

    ///
    wrapper<int> a(42);
    use_wrapper(&a);

    std::cout << "\n\n";

    
    number<double> test = 32.2;
    std::cout << test.as<int>();
    return 0;
}