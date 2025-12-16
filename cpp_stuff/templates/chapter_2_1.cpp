#include <iostream>
#include <memory>
// i love this shit =))
template <typename... T>
void print(T... args)
{
    ((std::cout << args << ' '), ...);
}

template <int N>
void print_addition(int const &a)
{
    print(a + N);
}

template <typename T, size_t S>
class buffer
{
    T data[S];

public:
    // constexpr -> is a t compile time
    // T const* -> returns a pointer to the const TR so w cant modify the pointer. so its locking the pointer to not be modified?
    // const in the end -> constant memeber function we cant not modify it

    /// return pointer to const data , method is constant
    constexpr T const *get_data() const { return data; }

    // non constant version which allows modification
    constexpr T &operator[](size_t const index)
    {
        return data[index];
    }

    // read onlu method access
    constexpr T const &operator[](size_t const index) const
    {
        return data[index];
    }
};

struct device
{
    virtual void output() = 0;
    virtual ~device() {}
};

template <void(*action)()>
struct smart_device : device
{
    void output() override
    {
        (*action)();
    }
};

void say_hello_in_english()
{
    print("HELLO WORLD");
}
void say_hello_in_spanish()
{
    print("HOAL MUNDO");
}

int main()
{

    print(10, 2, "CATAAA  n7 ", 2, 2, 2);
    std::cout << "\n";
    print_addition<10>(10);

    std::cout << "\n";
    std::cout << "\n";
    buffer<int, 10> b1;
    buffer<int, 10> b2;

    static_assert(std::is_same_v<decltype(b1), decltype(b2)>);

    auto w1 = std::make_unique<smart_device<&say_hello_in_english>>();
    w1->output();

    auto w2 = std::make_unique<smart_device<&say_hello_in_spanish>>();
    w2->output();
}